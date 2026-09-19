package org.sih26.deadreckoning.replay

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import kotlin.math.cos
import kotlin.math.sin
import org.sih26.deadreckoning.fusion.LocalFrame
import org.sih26.deadreckoning.fusion.MountLeveling
import org.sih26.deadreckoning.fusion.Stage12Calibration
import org.sih26.deadreckoning.fusion.Stage12Config
import org.sih26.deadreckoning.fusion.Stage12Decimator
import org.sih26.deadreckoning.fusion.Stage12FeatureExtractor
import org.sih26.deadreckoning.fusion.Stage12SpeedModel
import org.sih26.deadreckoning.fusion.Stage12Window

/**
 * Reads the two things the Replay page can play:
 *
 *  - a phone recording in this app's own JSONL format (`SessionLogger`), whether it
 *    came from the Sessions list or was exported from another phone, and
 *  - an IO-VNBD smartphone ("S-") CSV.
 *
 * The format is sniffed from the first line, so callers never have to say which.
 *
 * The JSONL parser looks at `gnss`, `fused`, `coast` and (when [Stage12SpeedModel] is
 * supplied) `imu` lines. Numbers are pulled out by key rather than through a JSON
 * library because the logger writes a fixed, flat schema and several of these streams
 * are 100 Hz each - allocating a JSON object per line to read a handful of doubles
 * would be most of the load time.
 *
 * Failures throw [IllegalArgumentException] with a message meant to be shown to the
 * person as-is.
 */
object ReplayLoader {

    /** Fused/coast are logged every IMU cycle (100 Hz); the trajectory view has no use
     * for more than this many points per second. */
    private const val MIN_TRACK_SPACING_S = 0.2

    /** [stage12Model] is optional. For a phone-session JSONL it drives
     * [buildStage12Trajectory] over that file's raw `imu` lines to add a second,
     * GNSS-independent trajectory to the returned session. For an IO-VNBD CSV it drives
     * [IovnbdStage12.build] over the file's own 10 Hz accelerometer/gyroscope columns,
     * producing a trajectory that follows GNSS except through simulated blackouts (see
     * [ReplaySession.blackouts]). Null (the default) skips all of this and behaves
     * exactly as before Stage 12 replay existed. */
    fun load(input: InputStream, label: String, stage12Model: Stage12SpeedModel? = null): ReplaySession {
        val reader = BufferedReader(InputStreamReader(input, Charsets.UTF_8), 1 shl 16)
        reader.mark(1 shl 16)
        var first = reader.readLine()
        while (first != null && first.isBlank()) first = reader.readLine()
        require(first != null) { "The file is empty." }
        reader.reset()

        return if (first.trimStart('\uFEFF', ' ').startsWith("{")) {
            parseJsonl(reader, label, stage12Model)
        } else {
            parseIovnbdCsv(reader, label, stage12Model)
        }
    }

    // ---------------------------------------------------------------- phone JSONL

    private const val GNSS_PREFIX = "{\"type\":\"gnss\""
    private const val FUSED_PREFIX = "{\"type\":\"fused\""
    private const val COAST_PREFIX = "{\"type\":\"coast\""
    private const val IMU_PREFIX = "{\"type\":\"imu\""
    private const val T = "\"t\":"
    private const val PN = "\"pn\":"
    private const val PE = "\"pe\":"
    private const val LAT = "\"lat\":"
    private const val LON = "\"lon\":"
    private const val SPEED = "\"speed\":"
    private const val BEARING = "\"bearing\":"
    private const val ACCURACY = "\"accuracy\":"
    private const val AX = "\"ax\":"
    private const val AY = "\"ay\":"
    private const val AZ = "\"az\":"
    private const val GX = "\"gx\":"
    private const val GY = "\"gy\":"
    private const val GZ = "\"gz\":"

    private data class ImuSample(
        val t: Double,
        val ax: Double, val ay: Double, val az: Double,
        val gx: Double, val gy: Double, val gz: Double
    )

    private fun parseJsonl(reader: BufferedReader, label: String, stage12Model: Stage12SpeedModel?): ReplaySession {
        var frame: LocalFrame? = null
        val truth = ArrayList<TruthFix>()
        val fused = ArrayList<ReplayPoint>()
        val coast = ArrayList<ReplayPoint>()
        // Only collected when a model was actually supplied - a plain replay (no
        // Stage 12 asset available) should not pay to buffer every 100 Hz line.
        val imu = if (stage12Model != null) ArrayList<ImuSample>() else null
        var lastFusedT = Double.NEGATIVE_INFINITY
        var lastCoastT = Double.NEGATIVE_INFINITY

        while (true) {
            val line = reader.readLine() ?: break
            when {
                line.startsWith(GNSS_PREFIX) -> {
                    val t = number(line, T) ?: continue
                    val lat = number(line, LAT) ?: continue
                    val lon = number(line, LON) ?: continue
                    // The first fix in the log is the frame origin: it is the first one
                    // the recording pipeline saw, and its fused/coast lines are relative
                    // to that same point.
                    val f = frame ?: LocalFrame(lat, lon).also { frame = it }
                    val ne = f.toNorthEast(lat, lon)
                    truth.add(
                        TruthFix(
                            t = t, north = ne[0], east = ne[1],
                            speedMps = number(line, SPEED),
                            accuracyM = number(line, ACCURACY),
                            headingDeg = number(line, BEARING),
                            satellites = null,
                            withheld = line.contains("\"withheld\":true")
                        )
                    )
                }
                line.startsWith(FUSED_PREFIX) -> {
                    val t = number(line, T) ?: continue
                    if (t - lastFusedT < MIN_TRACK_SPACING_S) continue
                    val n = number(line, PN) ?: continue
                    val e = number(line, PE) ?: continue
                    lastFusedT = t
                    fused.add(ReplayPoint(t, n, e, ReplayTrack.FUSED))
                }
                line.startsWith(COAST_PREFIX) -> {
                    val t = number(line, T) ?: continue
                    if (t - lastCoastT < MIN_TRACK_SPACING_S) continue
                    val n = number(line, PN) ?: continue
                    val e = number(line, PE) ?: continue
                    lastCoastT = t
                    coast.add(ReplayPoint(t, n, e, ReplayTrack.COAST))
                }
                imu != null && line.startsWith(IMU_PREFIX) -> {
                    val t = number(line, T) ?: continue
                    val ax = number(line, AX) ?: continue
                    val ay = number(line, AY) ?: continue
                    val az = number(line, AZ) ?: continue
                    val gx = number(line, GX) ?: continue
                    val gy = number(line, GY) ?: continue
                    val gz = number(line, GZ) ?: continue
                    imu.add(ImuSample(t, ax, ay, az, gx, gy, gz))
                }
                // header, or imu when no model was supplied: not needed here.
            }
        }

        require(truth.isNotEmpty()) { "No GNSS fixes in this session, so there is no track to replay." }

        // Stage 12's own trajectory, independent of the fused/coast the recording
        // pipeline already computed - see the function doc for what "independent"
        // means here. Never allowed to fail the whole import: a model that cannot
        // produce a usable track for this file still leaves truth/fused/coast intact.
        val stage12 = if (imu != null && stage12Model != null) {
            runCatching { buildStage12Trajectory(imu, truth, stage12Model) }.getOrElse { emptyList() }
        } else {
            emptyList()
        }

        val starts = listOfNotNull(truth.first().t, fused.firstOrNull()?.t, coast.firstOrNull()?.t)
        val ends = listOfNotNull(truth.last().t, fused.lastOrNull()?.t, coast.lastOrNull()?.t)
        return ReplaySession(
            label = label, source = ReplaySource.PHONE_SESSION,
            startS = starts.min(), endS = ends.max(),
            truth = truth, fused = fused, coast = coast, stage12 = stage12,
            originLat0Deg = frame?.lat0Deg, originLon0Deg = frame?.lon0Deg
        )
    }

    /** Builds a second, GNSS-independent trajectory from this file's raw IMU stream:
     * Stage 12's calibrated speed and corrected yaw rate, dead-reckoned forward from
     * the session's first fix (its position and heading only - not its subsequent
     * fixes), exactly the way [org.sih26.deadreckoning.fusion.Stage12Channel] and
     * [org.sih26.deadreckoning.fusion.PhysicsSpeedChannel] integrate live.
     *
     * Unlike the live pipeline (which only lets Stage 12 drive position during a real
     * blackout, gated by [org.sih26.deadreckoning.fusion.Stage12Channel.update]'s
     * `blackout` parameter), this always uses Stage 12's prediction: the point of a
     * replay import is to see what the model alone would have produced for the whole
     * drive, not to reproduce the live blackout-gating decision.
     *
     * Calibration (raw model speed -> GNSS-scale speed) is fit once, across every
     * (predicted, GNSS) pair the whole file offers, then frozen and applied
     * throughout - a whole-file fit rather than the live pipeline's strictly
     * pre-blackout-only one, since there is no blackout boundary in an import.
     */
    private fun buildStage12Trajectory(
        imu: List<ImuSample>,
        truth: List<TruthFix>,
        model: Stage12SpeedModel
    ): List<ReplayPoint> {
        if (imu.size < 50) return emptyList()

        // The app's own operating assumption ("Start parked so leveling can find
        // gravity") holds for any session it recorded, so the first slice of imu
        // doubles as the stationary window MountLeveling needs.
        val levelWindowCount = minOf(300, imu.size / 4).coerceAtLeast(2)
        val leveling = MountLeveling.fromStationaryWindow(
            imu.take(levelWindowCount).map { doubleArrayOf(it.ax, it.ay, it.az) }
        )

        val extractor = Stage12FeatureExtractor(leveling)
        val config = Stage12Config()
        val decimator = Stage12Decimator(config)
        val window = Stage12Window(config)
        val calibration = Stage12Calibration(config)

        data class Tick(val t: Double, val speedMps: Double, val yawRateRadS: Double)

        val ticks = ArrayList<Tick>()
        var truthIdx = 0

        for (s in imu) {
            val raw = extractor.extract(doubleArrayOf(s.ax, s.ay, s.az), doubleArrayOf(s.gx, s.gy, s.gz))
            val tNs = (s.t * 1_000_000_000.0).toLong()
            val decimated = decimator.add(tNs, raw) ?: continue
            val normalized = extractor.normalize(decimated)
            window.push(normalized)
            val (accWindow, gyroWindow) = window.toModelInput() ?: continue
            val prediction = model.predict(accWindow, gyroWindow) ?: continue

            while (truthIdx + 1 < truth.size && truth[truthIdx + 1].t <= s.t) truthIdx++
            val gnssSpeed = truth[truthIdx].speedMps
            if (gnssSpeed != null) calibration.observe(prediction.speedMps, gnssSpeed)

            val yawRate = leveling.yawRate(doubleArrayOf(s.gx, s.gy, s.gz)) + prediction.yawRateCorrectionRadS
            ticks.add(Tick(s.t, prediction.speedMps, yawRate))
        }
        if (ticks.isEmpty()) return emptyList()
        calibration.freeze()

        var north = truth.first().north
        var east = truth.first().east
        var heading = Math.toRadians(truth.first().headingDeg ?: 0.0)
        var lastT = ticks.first().t
        val out = ArrayList<ReplayPoint>(ticks.size)
        out.add(ReplayPoint(lastT, north, east, ReplayTrack.STAGE12))
        for (i in 1 until ticks.size) {
            val tick = ticks[i]
            val dt = (tick.t - lastT).coerceIn(0.0, 1.0)
            heading += tick.yawRateRadS * dt
            val speed = calibration.apply(tick.speedMps)
            // North/east convention matches fusion/Geo.kt: heading measured from
            // north, vn = speed*cos(heading), ve = speed*sin(heading).
            north += speed * cos(heading) * dt
            east += speed * sin(heading) * dt
            lastT = tick.t
            out.add(ReplayPoint(lastT, north, east, ReplayTrack.STAGE12))
        }
        return out
    }

    /** Value after [marker] up to the next comma or closing brace, or null if the key is
     * absent or not a finite number. */
    private fun number(line: String, marker: String): Double? {
        val at = line.indexOf(marker)
        if (at < 0) return null
        val start = at + marker.length
        var end = start
        while (end < line.length && line[end] != ',' && line[end] != '}') end++
        return line.substring(start, end).trim().toDoubleOrNull()?.takeIf { it.isFinite() }
    }

    // ------------------------------------------------------------- IO-VNBD S-CSV

    // Matched by header keyword rather than position, the same approach as
    // data/scripts/iovnbd_common.py's S_COLUMNS, so a small header spelling difference
    // does not silently read the wrong column. Patterns are against the lower-cased,
    // whitespace-collapsed header.
    private val TIME_COL = Regex("time\\s*since\\s*start")
    private val LAT_COL = Regex("gps\\s*latitude")
    private val LON_COL = Regex("gps\\s*longitude")
    private val SPEED_COL = Regex("gps\\s*speed")
    private val ACCURACY_COL = Regex("gps\\s*accuracy")
    private val ORIENTATION_COL = Regex("gps\\s*orientation")
    private val SATELLITES_COL = Regex("satellites\\s*in\\s*range")
    private val WHITESPACE = Regex("\\s+")

    private val ACC_X_COL = Regex("accelerometer\\s*x")
    private val ACC_Y_COL = Regex("accelerometer\\s*y")
    private val ACC_Z_COL = Regex("accelerometer\\s*z")

    /** Some IO-VNBD files name the gyro columns X/Y/Z and others Yaw/Pitch/Roll; either
     * way MotionSpeedNet's gx/gy/gz are simply the three gyroscope columns in file order
     * (`tools/baseline/data_loader.py`), so prefer explicit X/Y/Z names and otherwise
     * take the first three columns that mention the gyroscope. */
    private fun findGyroColumns(headers: List<String>): List<Int> {
        val byAxis = listOf("x", "y", "z").map { axis ->
            headers.indexOfFirst { Regex("gyroscope\\s*$axis\\b").containsMatchIn(it) }
        }
        if (byAxis.all { it >= 0 }) return byAxis
        return headers.indices.filter { headers[it].contains("gyroscope") }.take(3)
    }

    private fun parseIovnbdCsv(reader: BufferedReader, label: String, stage12Model: Stage12SpeedModel?): ReplaySession {
        var headerLine = reader.readLine()
        while (headerLine != null && headerLine.isBlank()) headerLine = reader.readLine()
        require(headerLine != null) { "The file is empty." }

        val headers = splitCsv(headerLine.trimStart('\uFEFF')).map { it.trim().lowercase().replace(WHITESPACE, " ") }
        fun find(pattern: Regex) = headers.indexOfFirst { pattern.containsMatchIn(it) }

        val timeIdx = find(TIME_COL)
        val latIdx = find(LAT_COL)
        val lonIdx = find(LON_COL)
        val missing = buildList {
            if (timeIdx < 0) add("Time since start")
            if (latIdx < 0) add("GPS Latitude")
            if (lonIdx < 0) add("GPS Longitude")
        }
        require(missing.isEmpty()) {
            "This does not look like an IO-VNBD smartphone (S-) CSV - missing column(s): " +
                missing.joinToString(", ") + "."
        }
        val speedIdx = find(SPEED_COL)
        val accuracyIdx = find(ACCURACY_COL)
        val orientationIdx = find(ORIENTATION_COL)
        val satellitesIdx = find(SATELLITES_COL)

        val accCols = listOf(find(ACC_X_COL), find(ACC_Y_COL), find(ACC_Z_COL))
        val gyroCols = findGyroColumns(headers)
        val hasImuColumns = accCols.all { it >= 0 } && gyroCols.size == 3
        // Only buffered when a model was supplied, same reasoning as the JSONL path.
        val imuRows = if (stage12Model != null && hasImuColumns) ArrayList<ImuRow>() else null

        val truth = ArrayList<TruthFix>()
        var frame: LocalFrame? = null
        var rawOriginS = 0.0
        var prevRawS = Double.NaN
        var clockOffsetS = 0.0
        var lastRowT = 0.0
        var lastImuT = Double.NEGATIVE_INFINITY
        var lastLat = Double.NaN
        var lastLon = Double.NaN

        while (true) {
            val line = reader.readLine() ?: break
            if (line.isBlank()) continue
            val f = splitCsv(line)
            fun cell(i: Int): Double? =
                if (i < 0) null else f.getOrNull(i)?.trim()?.toDoubleOrNull()?.takeIf { it.isFinite() }

            val tMs = cell(timeIdx) ?: continue
            val tRaw = tMs / 1000.0
            if (prevRawS.isNaN()) {
                rawOriginS = tRaw
            } else if (tRaw < prevRawS) {
                // The phone's logger clock restarted mid-file (S-S2 has one). Carry on from
                // where the timeline had got to, one nominal 10 Hz row later, instead of
                // discarding everything after the reset - the same repair
                // tools/preprocessing/reconstruct_timestamps.py applies for training.
                clockOffsetS = lastRowT + 0.1 - (tRaw - rawOriginS)
            }
            prevRawS = tRaw
            val t = (tRaw - rawOriginS) + clockOffsetS
            lastRowT = t

            if (imuRows != null && t > lastImuT) {
                val ax = cell(accCols[0])
                val ay = cell(accCols[1])
                val az = cell(accCols[2])
                val gx = cell(gyroCols[0])
                val gy = cell(gyroCols[1])
                val gz = cell(gyroCols[2])
                if (ax != null && ay != null && az != null && gx != null && gy != null && gz != null) {
                    // IO-VNBD logs GPS speed in km/h.
                    imuRows.add(ImuRow(t, ax, ay, az, gx, gy, gz, cell(speedIdx)?.let { it / 3.6 }))
                    lastImuT = t
                }
            }

            val lat = cell(latIdx) ?: continue
            val lon = cell(lonIdx) ?: continue
            if (lat == 0.0 && lon == 0.0) continue
            // The file carries GPS columns on every (much faster) IMU row; keep a point
            // only when the position actually changed, which recovers the ~1 Hz fixes.
            if (lat == lastLat && lon == lastLon) continue
            lastLat = lat
            lastLon = lon

            val fr = frame ?: LocalFrame(lat, lon).also { frame = it }
            val ne = fr.toNorthEast(lat, lon)
            truth.add(
                TruthFix(
                    t = t, north = ne[0], east = ne[1],
                    // IO-VNBD logs GPS speed in km/h.
                    speedMps = cell(speedIdx)?.let { it / 3.6 },
                    accuracyM = cell(accuracyIdx),
                    headingDeg = cell(orientationIdx),
                    satellites = cell(satellitesIdx)?.toInt(),
                    withheld = false
                )
            )
        }

        require(truth.isNotEmpty()) { "No usable GPS fixes in this file (GPS Latitude / GPS Longitude were empty or zero)." }

        // Stage 12's second trajectory, with simulated GNSS blackouts - the slow part of an
        // import. Never allowed to fail it: a model that cannot produce a track for this
        // file still leaves the GNSS replay intact.
        val stage12 = if (imuRows != null && stage12Model != null) {
            runCatching { IovnbdStage12.build(imuRows, truth, stage12Model) }.getOrNull()
        } else {
            null
        }

        return ReplaySession(
            label = label, source = ReplaySource.IOVNBD_CSV,
            startS = 0.0, endS = maxOf(lastRowT, truth.last().t),
            truth = truth, fused = emptyList(), coast = emptyList(),
            stage12 = stage12?.points ?: emptyList(),
            blackouts = stage12?.blackouts ?: emptyList(),
            originLat0Deg = frame?.lat0Deg, originLon0Deg = frame?.lon0Deg
        )
    }

    /** Comma split that respects double-quoted fields (some IO-VNBD headers contain
     * commas inside quotes). No escaped-quote handling: nothing in this dataset uses it. */
    private fun splitCsv(line: String): List<String> {
        val out = ArrayList<String>()
        val cell = StringBuilder()
        var inQuotes = false
        for (c in line) {
            when {
                c == '"' -> inQuotes = !inQuotes
                c == ',' && !inQuotes -> {
                    out.add(cell.toString())
                    cell.setLength(0)
                }
                else -> cell.append(c)
            }
        }
        out.add(cell.toString())
        return out
    }
}
