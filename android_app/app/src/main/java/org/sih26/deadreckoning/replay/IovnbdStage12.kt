package org.sih26.deadreckoning.replay

import java.util.Random
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt
import org.sih26.deadreckoning.fusion.Stage12Calibration
import org.sih26.deadreckoning.fusion.Stage12Config
import org.sih26.deadreckoning.fusion.Stage12NormStats
import org.sih26.deadreckoning.fusion.Stage12SpeedModel
import org.sih26.deadreckoning.fusion.Stage12Window

/**
 * One row of an IO-VNBD smartphone (S-) CSV's own inertial columns, on the session clock
 * ([t], seconds, already repaired for logger clock resets by [ReplayLoader]).
 *
 * The accelerometer still contains gravity (m/s^2) and the gyro is rad/s, both exactly as
 * the file stores them. [gx], [gy], [gz] are the file's three gyroscope columns in the
 * order they appear - the same slots `tools/baseline/data_loader.py` feeds MotionSpeedNet
 * as gx/gy/gz - and it is [gy] that doubles as the yaw rate, as in
 * `canonical_evaluate_v4.py` (`aligned_yaw_rate = df['gy']`).
 */
data class ImuRow(
    val t: Double,
    val ax: Double, val ay: Double, val az: Double,
    val gx: Double, val gy: Double, val gz: Double,
    /** GPS speed carried on the same row, m/s. Repeats between the ~1 Hz fixes. */
    val gpsSpeedMps: Double?
)

/**
 * How the simulated GNSS outages are laid out over a drive. Deterministic for a given
 * [seed] and drive length, so the same file always replays with the same blackouts and
 * two runs can be compared.
 */
data class BlackoutConfig(
    val minDurationS: Double = 20.0,
    val maxDurationS: Double = 50.0,
    /** Clean GNSS time between the end of one outage and the start of the next. Must
     * comfortably exceed the 30 s of pre-blackout data the speed calibration is fit on
     * ([Stage12Config.calibBufferSize] at 10 Hz). */
    val minGapS: Double = 60.0,
    val maxGapS: Double = 120.0,
    /** No outage starts before this long into the drive, so the model has a full
     * calibration window of GNSS-on driving first. */
    val firstStartAfterS: Double = 60.0,
    /** An outage where the vehicle was (per GNSS) hardly moving is skipped: dead
     * reckoning a parked car proves nothing. */
    val minMeanSpeedMps: Double = 1.0,
    val seed: Long = 26168L
)

/**
 * Builds the second trajectory of an IO-VNBD replay: what Stage 12 would have produced if
 * GNSS had been lost for 20-50 s at intervals along the drive.
 *
 * Runs at import, on the file's own 10 Hz inertial rows - which is MotionSpeedNet's
 * training cadence, so unlike a phone session there is no decimation step - and reproduces
 * `tools/baseline/data_loader.py`'s feature construction (EMA gravity removal with
 * alpha 0.05, magnitudes, the same normalisation constants) and
 * `canonical_evaluate_v4.py`'s inference loop (a causal 50-tick window that ends *before*
 * the sample being predicted, edge-padded at the start). Nothing here needs the file
 * pre-processed.
 *
 * The rule for the track, row by row:
 *  - GNSS available: Stage 12 is fed the fix, so its track is the GNSS track and sits on
 *    top of it. The speed calibration observes (model speed, GNSS speed) pairs the whole
 *    time.
 *  - GNSS lost: the calibration is frozen on the last ~30 s of GNSS-on data, the heading is
 *    taken from the last few seconds of fixes, and position is dead-reckoned from the last
 *    fix on the calibrated model speed and the gyro yaw rate plus the model's yaw-rate
 *    correction - the same speed/yaw sources [ReplayLoader]'s phone path and the live
 *    [org.sih26.deadreckoning.fusion.Stage12Channel] use.
 *  - GNSS back: the track snaps to the first returning fix. The jump in the drawn line is
 *    the accumulated drift, made visible.
 *
 * Deliberately not included: the offline benchmark's road-corridor filter and progress
 * anchors. Those need an OSM road polyline; this is the model's speed and yaw alone, which
 * makes the divergence numbers an honest (pessimistic) view of the model rather than of the
 * map.
 */
object IovnbdStage12 {

    class Output(val points: List<ReplayPoint>, val blackouts: List<BlackoutWindow>)

    /** Dead-reckoned points are thinned to this spacing; the trajectory view has no use
     * for the full 10 Hz. */
    private const val MIN_POINT_SPACING_S = 0.2

    /** Extra data before an outage start that is run through the model, so the
     * calibration buffer is full when it is frozen. */
    private const val CALIBRATION_LEAD_S = 32.0

    /** Window over which the pre-outage heading is measured from GNSS fixes. */
    private const val HEADING_LOOKBACK_S = 6.0
    private const val HEADING_MIN_DISTANCE_M = 5.0

    private const val GRAVITY_EMA_ALPHA = 0.05

    fun build(
        rows: List<ImuRow>,
        truth: List<TruthFix>,
        model: Stage12SpeedModel,
        blackoutConfig: BlackoutConfig = BlackoutConfig(),
        config: Stage12Config = Stage12Config()
    ): Output {
        if (rows.size < 60 || truth.size < 2) return Output(emptyList(), emptyList())

        val schedule = scheduleBlackouts(rows.first().t, rows.last().t, blackoutConfig)
            .filter { (s, e) -> movedEnough(truth, s, e, blackoutConfig.minMeanSpeedMps) }

        val n = rows.size
        val predSpeed = DoubleArray(n)
        val predYawCorr = DoubleArray(n)
        val predicted = BooleanArray(n)

        // ---- Pass 1: features and model inference --------------------------------------
        // Only rows that can matter are sent to the model (each outage plus the lead-in the
        // calibration is fit on); the window still sees every row so its history is intact.
        val window = Stage12Window(config)
        var gravX = 0.0
        var gravY = 0.0
        var gravZ = 0.0
        for (i in 0 until n) {
            val r = rows[i]
            if (i == 0) {
                gravX = r.ax; gravY = r.ay; gravZ = r.az
            } else {
                gravX = GRAVITY_EMA_ALPHA * r.ax + (1 - GRAVITY_EMA_ALPHA) * gravX
                gravY = GRAVITY_EMA_ALPHA * r.ay + (1 - GRAVITY_EMA_ALPHA) * gravY
                gravZ = GRAVITY_EMA_ALPHA * r.az + (1 - GRAVITY_EMA_ALPHA) * gravZ
            }
            val lx = r.ax - gravX
            val ly = r.ay - gravY
            val lz = r.az - gravZ

            if (needsModel(r.t, schedule)) {
                val input = window.toModelInput()
                if (input != null) {
                    val p = model.predict(input.first, input.second)
                    if (p != null) {
                        predSpeed[i] = p.speedMps
                        predYawCorr[i] = p.yawRateCorrectionRadS
                        predicted[i] = true
                    }
                }
            }

            val raw = doubleArrayOf(
                lx, ly, lz, sqrt(lx * lx + ly * ly + lz * lz),
                r.gx, r.gy, r.gz, sqrt(r.gx * r.gx + r.gy * r.gy + r.gz * r.gz)
            )
            window.push(Stage12NormStats.normalize(raw))
        }

        // ---- Pass 2: the track ---------------------------------------------------------
        val points = ArrayList<ReplayPoint>()
        val outages = ArrayList<BlackoutWindow>()
        val calibration = Stage12Calibration(config)

        var fixIdx = 0
        var north = truth.first().north
        var east = truth.first().east
        var heading = Math.toRadians(truth.first().headingDeg ?: 0.0)
        var prevT = rows.first().t

        var activeWindow = -1
        var windowPoints = ArrayList<ReplayPoint>()
        var lastEmitT = Double.NEGATIVE_INFINITY
        var lastRawSpeed = 0.0
        var lastYawCorr = 0.0
        var haveModelOutput = false

        fun closeOutage(index: Int) {
            val (s, e) = schedule[index]
            outages.add(measureOutage(truth, s, e, windowPoints))
            windowPoints = ArrayList()
        }

        for (i in 0 until n) {
            val r = rows[i]
            val dt = (r.t - prevT).coerceIn(0.0, 1.0)
            prevT = r.t

            val inWindow = schedule.indexOfFirst { (s, e) -> r.t >= s && r.t < e }

            if (inWindow < 0) {
                if (activeWindow >= 0) {
                    closeOutage(activeWindow)
                    activeWindow = -1
                }
                // GNSS is feeding Stage 12: its position is the fix.
                while (fixIdx < truth.size && truth[fixIdx].t <= r.t) {
                    val f = truth[fixIdx]
                    points.add(ReplayPoint(f.t, f.north, f.east, ReplayTrack.STAGE12))
                    north = f.north
                    east = f.east
                    fixIdx++
                }
                val gnssSpeed = r.gpsSpeedMps
                if (predicted[i] && gnssSpeed != null) calibration.observe(predSpeed[i], gnssSpeed)
                continue
            }

            // ---- inside a simulated outage: fixes are consumed but withheld -------------
            while (fixIdx < truth.size && truth[fixIdx].t <= r.t) fixIdx++

            if (inWindow != activeWindow) {
                if (activeWindow >= 0) closeOutage(activeWindow)
                activeWindow = inWindow
                calibration.freeze()
                haveModelOutput = false
                val lastFix = truth.getOrNull(fixIdx - 1)
                if (lastFix != null) {
                    north = lastFix.north
                    east = lastFix.east
                    heading = headingBeforeOutage(truth, fixIdx - 1)
                }
            }

            if (predicted[i]) {
                lastRawSpeed = predSpeed[i]
                lastYawCorr = predYawCorr[i]
                haveModelOutput = true
            }
            // Until the model has produced anything, hold the last GNSS speed rather than
            // invent one; in practice the lead-in makes this a zero-length case.
            val speed = if (haveModelOutput) {
                calibration.apply(lastRawSpeed)
            } else {
                truth.getOrNull(fixIdx - 1)?.speedMps ?: 0.0
            }
            val yawRate = r.gy + (if (haveModelOutput) lastYawCorr else 0.0)

            // Heading is measured from north, positive toward east (fusion/Geo.kt), and
            // the yaw rate is added to it exactly as the offline benchmark's UKF does.
            heading += yawRate * dt
            north += speed * cos(heading) * dt
            east += speed * sin(heading) * dt

            if (r.t - lastEmitT >= MIN_POINT_SPACING_S) {
                val p = ReplayPoint(r.t, north, east, ReplayTrack.STAGE12)
                points.add(p)
                windowPoints.add(p)
                lastEmitT = r.t
            }
        }
        if (activeWindow >= 0) closeOutage(activeWindow)

        return Output(points, outages)
    }

    /** Outage start/end times over [startS, endS], from [BlackoutConfig]'s rules. */
    internal fun scheduleBlackouts(startS: Double, endS: Double, cfg: BlackoutConfig): List<Pair<Double, Double>> {
        val rnd = Random(cfg.seed)
        fun between(lo: Double, hi: Double) = lo + rnd.nextDouble() * (hi - lo)

        val out = ArrayList<Pair<Double, Double>>()
        var cursor = startS + cfg.firstStartAfterS
        var first = true
        while (true) {
            // The first outage may start right at the warm-up boundary; later ones wait a
            // full clean-GNSS gap after the previous outage ends.
            val gap = if (first) between(0.0, cfg.maxGapS - cfg.minGapS) else between(cfg.minGapS, cfg.maxGapS)
            val start = cursor + gap
            val end = start + between(cfg.minDurationS, cfg.maxDurationS)
            if (end > endS) break
            out.add(Pair(start, end))
            cursor = end
            first = false
        }
        return out
    }

    private fun needsModel(t: Double, schedule: List<Pair<Double, Double>>): Boolean =
        schedule.any { (s, e) -> t >= s - CALIBRATION_LEAD_S && t < e }

    /** True unless GNSS says the vehicle was essentially parked for the whole window. */
    private fun movedEnough(truth: List<TruthFix>, startS: Double, endS: Double, minMeanSpeedMps: Double): Boolean {
        var sum = 0.0
        var count = 0
        for (f in truth) {
            if (f.t < startS) continue
            if (f.t >= endS) break
            val v = f.speedMps ?: continue
            sum += v
            count++
        }
        return count == 0 || sum / count >= minMeanSpeedMps
    }

    /** Heading (radians from north) going into an outage: the direction of travel over the
     * last few seconds of fixes when the vehicle moved far enough for that to mean
     * something, else the fix's own reported bearing. */
    private fun headingBeforeOutage(truth: List<TruthFix>, lastFixIdx: Int): Double {
        val last = truth[lastFixIdx]
        var j = lastFixIdx
        while (j > 0 && last.t - truth[j - 1].t <= HEADING_LOOKBACK_S) j--
        val dn = last.north - truth[j].north
        val de = last.east - truth[j].east
        return if (hypot(dn, de) >= HEADING_MIN_DISTANCE_M) {
            atan2(de, dn)
        } else {
            Math.toRadians(last.headingDeg ?: 0.0)
        }
    }

    /** Drift metrics for one outage, against real GNSS interpolated to each
     * dead-reckoned point's own time. */
    private fun measureOutage(
        truth: List<TruthFix>,
        startS: Double,
        endS: Double,
        drPoints: List<ReplayPoint>
    ): BlackoutWindow {
        if (drPoints.isEmpty()) return BlackoutWindow(startS, endS, null, null, null)

        var maxError = 0.0
        var endError = 0.0
        for (p in drPoints) {
            val g = interpolateTruth(truth, p.t) ?: continue
            val e = hypot(p.north - g[0], p.east - g[1])
            if (e > maxError) maxError = e
            endError = e
        }

        // Distance the real vehicle covered while the outage lasted, from the fixes
        // inside it plus the interpolated positions at its two edges.
        val from = interpolateTruth(truth, startS)
        val to = interpolateTruth(truth, endS)
        var distance = 0.0
        if (from != null && to != null) {
            var pn = from[0]
            var pe = from[1]
            for (f in truth) {
                if (f.t <= startS) continue
                if (f.t >= endS) break
                distance += hypot(f.north - pn, f.east - pe)
                pn = f.north
                pe = f.east
            }
            distance += hypot(to[0] - pn, to[1] - pe)
        }
        return BlackoutWindow(startS, endS, endError, maxError, distance)
    }
}
