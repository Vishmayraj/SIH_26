package org.sih26.deadreckoning.fusion

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Kotlin bridge for `models/stage12/motion_speed_net.py` (MotionSpeedNet) and the
 * live-inference / calibration logic in `tools/stage12/canonical_evaluate_v4.py`
 * (`run_model_inference`, `calibrate_pre_blackout`). Deliberately Android-free, same
 * as the rest of this package, so the window/calibration math is testable on a
 * desktop JVM; the actual ONNX session lives in the sensors package, behind
 * [Stage12SpeedModel].
 *
 * What this bridges, and the three real differences from the offline benchmark that
 * had to be resolved rather than papered over:
 *
 * 1. Sample rate. MotionSpeedNet was trained on the IO-VNBD S-Dataset's native
 *    10 Hz cadence, 50-sample (5.0 s) causal windows. This app's IMU runs at 100 Hz
 *    (SessionRecordingService.SAMPLING_PERIOD_US). [Stage12Decimator] block-averages
 *    ten 100 Hz ticks into one 10 Hz tick before anything is normalised or windowed,
 *    both to land near the training cadence and, as an anti-aliasing side effect
 *    a naive every-10th-sample subsample would not give, to reduce the sensor
 *    noise a single 100 Hz sample carries.
 *
 * 2. Gravity removal. `canonical_evaluate_v4.py`'s features (`ax_body`, `ay_body`,
 *    `az_body`) come from `tools/baseline/data_loader.py`'s slow EMA gravity tracker
 *    (alpha=0.05) applied directly to the raw device-frame accelerometer - a proxy
 *    the training pipeline used because it had no per-session leveling estimate.
 *    This app already has a better one: [MountLeveling] resolves gravity exactly
 *    from a genuinely stationary window and holds it fixed for the session, which
 *    is the same fixed-mount assumption the EMA proxy was standing in for, done
 *    exactly rather than approximately. [Stage12FeatureExtractor] uses that instead
 *    of re-deriving a second, worse gravity estimate.
 *
 * 3. Axis convention - the single biggest domain-transfer risk here, flagged for
 *    physical-device validation rather than resolved with confidence. IO-VNBD's
 *    phones shared one fixed physical mount, and the model's per-axis
 *    normalisation stats (config/norm_stats.json) reflect it: `gy`'s std (0.202) is
 *    roughly double `gx`'s and `gz`'s (~0.10-0.11), the signature of a car that
 *    mostly only yaws and a mount whose Y axis happened to be the vertical one.
 *    `az_body`'s std (0.606) is roughly half of `ax_body`/`ay_body`'s (~1.16-1.30),
 *    the signature of a mostly-vertical residual against two horizontal,
 *    manoeuvre-driven axes. This app cannot assume any specific physical mount -
 *    that is the whole reason [MountLeveling] and [ForwardAxisEstimator] exist - so
 *    [Stage12FeatureExtractor] reproduces the *shape* of that convention using the
 *    leveled frame's own axes instead of the phone's raw ones: gravity-frame `up`
 *    into the `gy`/`ay_body`-standing-out slot, and the leveled frame's `x`/`y` into
 *    the other two. Which of the leveled x/y lands on `ax_body` vs `ay_body` (and
 *    `gx` vs `gz`) is arbitrary - [MountLeveling]'s own Gram-Schmidt seed has no
 *    relationship to IO-VNBD's mount - but is at least applied consistently between
 *    the accelerometer and gyroscope branches. A model trained on one fixed mount
 *    generalising to an axis-remapped-but-not-identical input distribution is an
 *    assumption, not a proven fact; see the handoff notes for what a physical test
 *    should check.
 */

/** Snapshot of `models/stage12/config/norm_stats.json`. Regenerate together with the
 * ONNX export if the model is ever retrained - these are frozen constants, not
 * loaded at runtime, so nothing here re-derives itself automatically from a new
 * checkpoint. See models/stage12/export.py. */
object Stage12NormStats {
    // acc slot order: [ax_body, ay_body, az_body, accel_mag]
    val accMean = doubleArrayOf(-4.5478985713665165e-05, 8.22056572064762e-06, -1.7415178420207325e-04, 1.3666331119846837)
    val accStd = doubleArrayOf(1.2963173920905586, 1.158747829854063, 0.6059120923129813, 1.2339266969442095)
    // gyro slot order: [gx, gy, gz, gyro_mag]
    val gyroMean = doubleArrayOf(-7.208285002185238e-04, -6.184151242041602e-03, -2.8332014316257024e-05, 0.16268821950902476)
    val gyroStd = doubleArrayOf(0.10156945587052349, 0.2020954589455439, 0.11273504824688856, 0.19349155417395442)

    /** Normalise a raw 8-vector `[ax, ay, az, accel_mag, gx, gy, gz, gyro_mag]` into the
     * model's `[acc(4), gyro(4)]` layout. Lives here rather than only on
     * [Stage12FeatureExtractor] so a caller that already has gravity-free features
     * (the IO-VNBD replay import, which reproduces `tools/baseline/data_loader.py`'s
     * own EMA gravity removal) can normalise without building a [MountLeveling]. */
    fun normalize(raw: DoubleArray): FloatArray {
        val out = FloatArray(8)
        for (i in 0 until 4) out[i] = ((raw[i] - accMean[i]) / accStd[i]).toFloat()
        for (i in 0 until 4) out[4 + i] = ((raw[4 + i] - gyroMean[i]) / gyroStd[i]).toFloat()
        return out
    }
}

data class Stage12Config(
    /** Causal window length in 10 Hz ticks. 50 = 5.0 s, matching MotionSpeedNet's
     * training window (`motion_speed_net.py`'s docstring, `canonical_evaluate_v4.py`'s
     * `window_len`). */
    val windowLen: Int = 50,
    /** Target period of one decimated model-rate tick, matching MotionSpeedNet's
     * 10 Hz training cadence (`tools/baseline/data_loader.py`'s native S-Dataset
     * rate). [Stage12Decimator] block-averages however many 100 Hz IMU ticks land
     * inside each period of this length - it does not assume the IMU is exactly
     * 100 Hz, only that it is fast enough to produce several per period. */
    val modelPeriodS: Double = 0.1,
    /** How many pre-blackout (predicted, GNSS) speed pairs to keep for the affine
     * calibration fit. 300 @ 10 Hz = 30 s, matching `calibrate_pre_blackout`'s
     * `calib_start = idx_start - 300`. */
    val calibBufferSize: Int = 300,
    /** Refuse to calibrate on fewer pairs than this. Not a concern in the offline
     * benchmark, which always has 30 s of pre-blackout driving by construction; a
     * live phone can black out earlier than that, and fitting a 2-point regression
     * is worse than not calibrating at all. */
    val minCalibSamples: Int = 30,
    val calibSlopeMin: Double = 0.5,
    val calibSlopeMax: Double = 2.0,
    val speedClampMin: Double = 0.0,
    val speedClampMax: Double = 40.0,
    /** Floor on the per-cycle 1-sigma handed to the UKF, matching
     * `canonical_evaluate_v4.py`'s `ukf.config.r_channel_a = max(0.25, u)`. */
    val minRChannelA: Double = 0.25
)

/** One model prediction, before calibration. Mirrors `MotionSpeedNet.forward`'s
 * output dict. `speedVariance` is `exp(speed_log_var)` - see the note on
 * [Stage12Output.rChannelA] for why it is handed to the UKF as if it were already a
 * 1-sigma rather than squared first; that is the Python reference's own convention,
 * reproduced deliberately rather than "corrected" out from under the validated
 * benchmark numbers it was measured against. */
data class Stage12Prediction(
    val speedMps: Double,
    val speedVariance: Double,
    val yawRateCorrectionRadS: Double
)

/** The ONNX inference boundary. Implemented against Android's ONNX Runtime session
 * in the sensors package; kept as an interface here so the window/calibration logic
 * above it is testable without an Android runtime or a real model file. */
interface Stage12SpeedModel {
    /** `accWindow`/`gyroWindow` are (windowLen, 4) causal, oldest-first, already
     * normalised. Returns null if the model failed to load or a single inference
     * call failed - the caller falls back to Channel P, never to a fabricated
     * prediction. */
    fun predict(accWindow: Array<FloatArray>, gyroWindow: Array<FloatArray>): Stage12Prediction?
}

/** Converts a leveled accel/gyro sample into MotionSpeedNet's 8-channel raw (not yet
 * normalised) feature layout. One instance per session: it is bound to a single
 * [MountLeveling], consistent with that leveling's own fixed-for-the-session
 * assumption (see the class doc, point 3). */
class Stage12FeatureExtractor(private val leveling: MountLeveling) {

    /** Returns [ax, ay, az, accel_mag, gx, gy, gz, gyro_mag], raw (not normalised).
     * `accel` and `gyro` are the same raw phone-frame samples [MountLeveling] and
     * [PhysicsSpeedChannel] already consume - TYPE_ACCELEROMETER m/s^2 with gravity
     * included, TYPE_GYROSCOPE rad/s. */
    fun extract(accel: DoubleArray, gyro: DoubleArray): DoubleArray {
        val horizontal = leveling.horizontal(accel)
        val up = leveling.up
        val verticalSpecificForce = LinAlg.dot(accel, up) - leveling.gravityMagnitude

        val ax = horizontal[0]
        val ay = horizontal[1]
        val az = verticalSpecificForce
        val accelMag = sqrt(ax * ax + ay * ay + az * az)

        val xLevel = leveling.rotation[0]
        val yLevel = leveling.rotation[1]
        val gx = LinAlg.dot(gyro, xLevel)
        val gy = leveling.yawRate(gyro) // dot(gyro, up)
        val gz = LinAlg.dot(gyro, yLevel)
        val gyroMag = LinAlg.norm(gyro)

        return doubleArrayOf(ax, ay, az, accelMag, gx, gy, gz, gyroMag)
    }

    /** Normalise a raw 8-vector from [extract] using [Stage12NormStats], splitting
     * it into the model's two 4-channel branches. */
    fun normalize(raw: DoubleArray): FloatArray = Stage12NormStats.normalize(raw)
}

/** Block-averages 100 Hz feature ticks down to ~10 Hz. Time-based rather than a
 * fixed every-Nth-sample counter, because "the phone will not hand you a clean
 * 10 ms cadence" (FusionPipeline's own onImu doc) applies here exactly as it does to
 * the UKF's dt handling - a fixed sample count drifts against wall-clock time if any
 * ticks are dropped or coalesced. */
class Stage12Decimator(private val config: Stage12Config = Stage12Config()) {
    private val targetPeriodNs = (config.modelPeriodS * 1_000_000_000.0).toLong()
    private var blockStartNs: Long? = null
    private var accum = DoubleArray(8)
    private var count = 0

    /** Feed one raw (unnormalised) 8-vector at IMU rate. Returns the block-averaged
     * 8-vector once a ~0.1 s block has been collected, else null. */
    fun add(tNs: Long, raw: DoubleArray): DoubleArray? {
        if (blockStartNs == null) blockStartNs = tNs
        for (i in 0 until 8) accum[i] += raw[i]
        count++

        val elapsed = tNs - blockStartNs!!
        if (elapsed < targetPeriodNs) return null

        val out = DoubleArray(8) { accum[it] / count }
        accum = DoubleArray(8)
        count = 0
        blockStartNs = tNs
        return out
    }
}

/** Causal ring buffer of the last [Stage12Config.windowLen] normalised 10 Hz ticks.
 * Edge-pads at the front while fewer than a full window has been collected,
 * matching `canonical_evaluate_v4.py`'s `np.pad(..., mode='edge')` for the first
 * 5 seconds of a run. */
class Stage12Window(private val config: Stage12Config = Stage12Config()) {
    private val ticks = ArrayDeque<FloatArray>()

    fun push(normalizedTick: FloatArray) {
        ticks.addLast(normalizedTick)
        while (ticks.size > config.windowLen) ticks.removeFirst()
    }

    val isEmpty: Boolean get() = ticks.isEmpty()

    /** (accWindow, gyroWindow), each (windowLen, 4), oldest-first. Null if nothing
     * has been pushed yet - there is no sample to edge-pad with. */
    fun toModelInput(): Pair<Array<FloatArray>, Array<FloatArray>>? {
        if (ticks.isEmpty()) return null
        val padCount = config.windowLen - ticks.size
        val first = ticks.first()

        val acc = Array(config.windowLen) { FloatArray(4) }
        val gyro = Array(config.windowLen) { FloatArray(4) }
        for (i in 0 until config.windowLen) {
            val tick = if (i < padCount) first else ticks[i - padCount]
            for (c in 0 until 4) {
                acc[i][c] = tick[c]
                gyro[i][c] = tick[4 + c]
            }
        }
        return Pair(acc, gyro)
    }
}

/** Snapshot affine calibration: `gps_speed ~= slope * predicted_speed + intercept`,
 * fit by ordinary least squares over the pre-blackout window and then frozen for the
 * duration of a blackout. Matches `calibrate_pre_blackout`'s affine path exactly;
 * its EWMA-bias path is not reproduced here (see [Stage12Channel]'s doc for why). */
class Stage12Calibration(private val config: Stage12Config = Stage12Config()) {
    private val predicted = ArrayDeque<Double>()
    private val truth = ArrayDeque<Double>()

    var slope: Double = 1.0
        private set
    var intercept: Double = 0.0
        private set
    var trusted: Boolean = false
        private set

    /** Call every tick while GNSS is available and the vehicle is not in a
     * blackout - this is the "pre-blackout GNSS-on data" the fit is strictly
     * limited to. */
    fun observe(predictedSpeed: Double, gnssSpeed: Double) {
        predicted.addLast(predictedSpeed)
        truth.addLast(gnssSpeed)
        while (predicted.size > config.calibBufferSize) {
            predicted.removeFirst()
            truth.removeFirst()
        }
    }

    /** Fit and freeze (slope, intercept) from everything observed so far. Call once,
     * at the moment a blackout starts - not repeated during the blackout, matching
     * the "strictly on pre-blackout data" contract. */
    fun freeze() {
        val n = predicted.size
        if (n < config.minCalibSamples) {
            slope = 1.0
            intercept = 0.0
            trusted = false
            return
        }

        var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0
        for (i in 0 until n) {
            val x = predicted[i]
            val y = truth[i]
            sx += x; sy += y; sxx += x * x; sxy += x * y
        }
        val denom = n * sxx - sx * sx
        if (abs(denom) < 1e-9) {
            slope = 1.0
            intercept = 0.0
            trusted = false
            return
        }
        val fittedSlope = (n * sxy - sx * sy) / denom
        slope = min(config.calibSlopeMax, max(config.calibSlopeMin, fittedSlope))
        intercept = (sy - slope * sx) / n
        trusted = true
    }

    fun apply(predictedSpeed: Double): Double {
        val calibrated = slope * predictedSpeed + intercept
        return min(config.speedClampMax, max(config.speedClampMin, calibrated))
    }
}

private fun abs(v: Double): Double = if (v < 0.0) -v else v

/** Output of one [Stage12Channel] cycle, ready to hand to [DualChannelUkf.step] in
 * Channel A's slot. */
data class Stage12Output(
    val calibratedSpeedMps: Double,
    /** `max(minRChannelA, exp(speed_log_var))`, passed straight through to
     * [DualChannelUkf.step]'s `rChannelAOverride`, which squares it into an R
     * itself. `exp(speed_log_var)` is mathematically a variance, not a 1-sigma, so
     * this is squared twice end to end - matching `canonical_evaluate_v4.py`'s own
     * `ukf.config.r_channel_a = max(0.25, u)` exactly rather than "fixing" an
     * apparent looseness the validated benchmark numbers were measured against. */
    val rChannelA: Double,
    val correctedYawRateRadS: Double
)

/**
 * Orchestrates extraction, decimation, windowing, model inference and calibration
 * for one session. One instance per [FusionPipeline] run, constructed once
 * [MountLeveling] is available.
 *
 * Runs continuously, independent of blackout state - the model keeps predicting and
 * the calibrator keeps collecting (GNSS-available, x) pairs the whole time, exactly
 * like `canonical_evaluate_v4.py`'s inference loop covers the pre-blackout
 * calibration window too. Whether that output actually reaches the UKF is
 * [FusionPipeline]'s call, not this class's: feeding an uncalibrated or
 * still-warming-up learned channel into the filter while GNSS is available and
 * already the best evidence in the room would only add noise for no benefit. See
 * [update]'s `blackout` parameter.
 */
class Stage12Channel(
    leveling: MountLeveling,
    private val model: Stage12SpeedModel,
    private val config: Stage12Config = Stage12Config()
) {
    private val extractor = Stage12FeatureExtractor(leveling)
    private val decimator = Stage12Decimator(config)
    private val window = Stage12Window(config)
    private val calibration = Stage12Calibration(config)

    private var lastRawPrediction: Stage12Prediction? = null

    /** Call once per IMU sample (100 Hz), in the same place [PhysicsSpeedChannel]
     * is driven. `leveledYawRate` is this cycle's own `MountLeveling.yawRate` (the
     * UKF steps at 100 Hz and needs a yaw rate every cycle regardless of the
     * model's 10 Hz refresh rate). `gnssSpeed` is the just-admitted fix's speed, or
     * null on a cycle with none; only meaningful (and only observed by the
     * calibrator) when `blackout` is false. Returns a channel-A-ready output only
     * when `blackout` is true and a calibrated prediction is available; null
     * otherwise, in which case the caller should fall back to (or continue using)
     * Channel P. */
    fun update(
        tNs: Long,
        accel: DoubleArray,
        gyro: DoubleArray,
        leveledYawRate: Double,
        gnssSpeed: Double?,
        blackout: Boolean
    ): Stage12Output? {
        val raw = extractor.extract(accel, gyro)
        val decimated = decimator.add(tNs, raw)
        if (decimated != null) {
            val normalized = extractor.normalize(decimated)
            window.push(normalized)

            val (accWindow, gyroWindow) = window.toModelInput()!!
            val prediction = model.predict(accWindow, gyroWindow)
            if (prediction != null) {
                lastRawPrediction = prediction
                if (!blackout && gnssSpeed != null) {
                    calibration.observe(prediction.speedMps, gnssSpeed)
                }
            }
        }

        return currentOutput(blackout, leveledYawRate)
    }

    /** Called once, at the instant [FusionPipeline] flips into blackout, so the
     * calibration fit is frozen on pre-blackout data only. */
    fun onBlackoutStart() {
        calibration.freeze()
    }

    private fun currentOutput(blackout: Boolean, leveledYawRate: Double): Stage12Output? {
        if (!blackout) return null
        val prediction = lastRawPrediction ?: return null
        if (!calibration.trusted) return null

        val calibratedSpeed = calibration.apply(prediction.speedMps)
        val rChannelA = max(config.minRChannelA, prediction.speedVariance)
        // yaw_rate_correction is trained against exactly the axis this extractor put
        // in the "gy" slot - the leveled frame's yaw rate - so it corrects that same
        // quantity here, not the phone's raw z axis.
        val correctedYawRate = leveledYawRate + prediction.yawRateCorrectionRadS
        return Stage12Output(calibratedSpeed, rChannelA, correctedYawRate)
    }
}
