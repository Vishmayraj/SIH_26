package org.sih26.deadreckoning.fusion

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max

/**
 * The online pipeline: raw phone sensors in, fused state out.
 *
 * Deliberately free of Android imports so the whole thing runs on a desktop JVM
 * against a recorded log. Every correctness question about this file can then be
 * answered without a device, which is the only reason the answers exist before the
 * drive rather than after it.
 *
 * Data flow, one direction, no cleverness:
 *
 *     onImu (100 Hz)  -> leveling -> horizontal accel + yaw rate
 *                     -> ZUPT detector
 *                     -> Channel P integration, Stage 12 model (if injected)
 *                     -> Corridor road-matching (if injected), reading whatever
 *                        speed/yaw Channel A settled on this cycle
 *                     -> UKF step
 *     onGnss (1 Hz)   -> staged for the next IMU cycle, or withheld if blacked out
 *                     -> also kicks off CorridorChannel's one-shot road-network fetch
 *
 * Stage 12 (models/stage12/motion_speed_net.py, bridged via [Stage12Channel]) only
 * takes Channel A's slot away from Channel P during a blackout, and only once its
 * pre-blackout calibration is trusted - see that class's doc for why.
 *
 * Corridor (tools/stage12/corridor_filter.py, bridged via [CorridorChannel]) tracks
 * along a road-network polyline through a blackout using the same Channel A speed
 * evidence, and only reaches the outer UKF as a position correction once its
 * progress-anchor confidence clears [FusionConfig.roadSignatureConfidenceThreshold]
 * - see that class's doc for the two-threshold reasoning.
 *
 * The blackout is a boolean gate on whether a fix reaches the filter. Fixes are
 * always received and always logged. That is the entire experiment: withheld fixes
 * are the ground truth the drift is measured against, and airplane mode would destroy
 * them and cost 10 to 30 seconds of reacquisition afterwards.
 */

/** A snapshot for the UI and for the optional fused stream in the session log. */
data class FusionSnapshot(
    val tSeconds: Double,
    val fusedNorth: Double,
    val fusedEast: Double,
    val coastNorth: Double,
    val coastEast: Double,
    val speedMps: Double,
    val headingDeg: Double,
    val blackout: Boolean,
    val blackoutElapsedS: Double,
    val blackoutDistanceM: Double,
    val driftMeters: Double,
    val driftPercent: Double,
    val channelPSpeed: Double?,
    /** Stage 12's own calibrated speed prediction this cycle, or null when the
     * model has not produced a trusted prediction yet (still warming up, or its
     * pre-blackout calibration never saw enough GNSS-available driving). Reported
     * regardless of [stage12Active] so a field test can see the model's raw
     * behavior even on cycles where Channel P was the one actually fed to the
     * UKF. */
    val stage12SpeedMps: Double?,
    /** True on cycles where [stage12SpeedMps] (not Channel P) was the value
     * actually fed into the UKF's Channel A slot - see Stage12Channel's class doc
     * for the blackout-gating rule. */
    val stage12Active: Boolean,
    /** True once [CorridorChannel] has a candidate road chained and is tracking
     * along it, independent of whether that track is currently confident enough
     * to reach the outer UKF (see [corridorActive]). False before any blackout,
     * after a blackout ends, or if the road-network fetch never completed / found
     * nothing nearby. */
    val corridorRoadLocked: Boolean,
    /** [RoadProgressMatcher]'s progress-anchor confidence this cycle, 0 if no
     * road is locked. Reported regardless of [corridorActive] so a field test can
     * see the match quality even on cycles below the outer UKF's own threshold. */
    val corridorConfidence: Double,
    /** True on cycles where the corridor filter's road-snapped position actually
     * reached the outer UKF as a position correction (confidence cleared
     * [FusionConfig.roadSignatureConfidenceThreshold]) - see the call site in
     * [onImu] for why this is a stricter gate than [corridorConfidence] alone. */
    val corridorActive: Boolean,
    /** Distance between the corridor filter's road-snapped position and the
     * outer UKF's own pre-correction position estimate this cycle - how hard the
     * road match is pulling, for display only. Null unless a road is locked on. */
    val corridorPositionCorrectionM: Double?,
    val zuptActive: Boolean,
    val imuRateHz: Double,
    val stepLatencyMs: Double,
    val phase: PipelinePhase,
    val lastTruthNorth: Double,
    val lastTruthEast: Double
)

enum class PipelinePhase {
    /** Collecting a stationary window so mount leveling has a gravity vector. Start
     * every session parked for a few seconds or this phase never completes cleanly. */
    LEVELING,

    /** Levelled, waiting for the first GNSS fix to set the local frame origin. */
    WAITING_FOR_GNSS,

    /** Filter initialised and running. */
    RUNNING
}

data class PipelineConfig(
    // gnssReacquireRMultiplierStart overridden from FusionConfig's own default of 50.0
    // (2500x variance) down to 5.0 (25x). At 50.0, GNSS position is blinded for the
    // whole 2.5 s reacquire ramp while a drifted Channel A speed keeps pulling
    // position on the stale heading, which is what produced the 60+ m/s reacquisition
    // loops in AndroidAppReadingsAnalysis.md. This is set here rather than in
    // FusionConfig's own default because tools/parity/generate_fixture.py and
    // VerifyParity.kt both instantiate FusionConfig() directly to check Kotlin against
    // the Python reference bit-for-bit; changing the shared default would silently
    // break that fixture. The pipeline every phone build actually runs is this class,
    // so overriding it here changes real behavior without touching parity.
    val fusion: FusionConfig = FusionConfig(gnssReacquireRMultiplierStart = 5.0),
    val physics: PhysicsSpeedConfig = PhysicsSpeedConfig(),
    val zupt: ZuptConfig = ZuptConfig(),
    /** Seconds of stationary samples required before leveling. */
    val levelingWindowS: Double = 3.0,
    /** Gyro magnitude above this during leveling means the phone is being held or
     * handled, not resting on the mount. Session 2 of AndroidAppReadingsAnalysis.md
     * started leveling while being handled (mean yaw rate 11.6 deg/s, peak 35 deg/s,
     * i.e. 0.2-0.61 rad/s) and the resulting "up" vector carried a 0.58 m/s^2
     * horizontal bias into every subsequent Channel P integration for the rest of the
     * session. A stationary phone's gyro noise floor sits well under this. */
    val levelingGyroQuietThresholdRadS: Double = 0.05,
    /** Bound on how long to keep restarting the leveling window in search of a quiet
     * one before giving up and leveling on whatever was collected anyway. Mirrors
     * maxWaitForMovingFixS below: a session that is handled throughout still has to
     * start eventually, but [levelingTrusted] says whether the result should be. */
    val maxLevelingWaitS: Double = 15.0,
    /** Speed above which a GNSS bearing is trusted for initial heading. At a
     * standstill the bearing is noise. */
    val minSpeedForBearingMps: Double = 1.5,
    /** How long to wait for a fix above that speed before giving up and seeding the
     * heading from whatever bearing is available. A session that never moves has no
     * dead reckoning to measure anyway, but the pipeline should still start and say
     * that its heading seed is untrusted rather than silently never initialise. */
    val maxWaitForMovingFixS: Double = 60.0,
    /** Channel P is fed to the filter as Channel A's slot with this override, since
     * its measured R is nothing like Channel A's Section 4.2 target. */
    val useChannelP: Boolean = true,
    val stage12: Stage12Config = Stage12Config(),
    /** Largest dt a single cycle may claim. A delivery hiccup or a resumed app must
     * not propagate the filter through a ten-second step as though it were one
     * sample. */
    val maxStepDtS: Double = 0.2
)

private const val NANOS_PER_SECOND = 1_000_000_000.0

/** Recompute the forward axis once a second rather than every cycle. The axis is a
 * property of the mount, so it does not move between recomputes, and the estimate is
 * linear in the accumulated sample count. */
private const val AXIS_RECOMPUTE_CYCLES = 100

class FusionPipeline(
    val config: PipelineConfig = PipelineConfig(),
    /** Optional Stage 12 speed model, injected rather than constructed here so this
     * file can stay Android-free (see the module doc at the top). Null disables the
     * channel entirely and the pipeline behaves exactly as before - Channel P is
     * the only Channel A source, same as prior to this parameter's existence. The
     * Android-side ONNX-backed implementation lives in the sensors package. */
    private val stage12Model: Stage12SpeedModel? = null,
    /** Optional road-network provider for [CorridorChannel] (Mode F: Best Adaptive
     * Hybrid). Null disables the channel entirely - same "injected, not
     * constructed" reasoning as [stage12Model], and the same degrade-to-prior-
     * behaviour guarantee: without it, this pipeline runs exactly as it did
     * before corridor matching existed. The Android-side Overpass-backed
     * implementation lives in the sensors package. */
    roadNetworkProvider: RoadNetworkProvider? = null
) {

    var phase: PipelinePhase = PipelinePhase.LEVELING
        private set

    var leveling: MountLeveling? = null
        private set
    var localFrame: LocalFrame? = null
        private set

    /** False when the filter had to seed its heading from a fix taken at or near a
     * standstill. Surface it: a session that starts this way has a heading the gyro
     * will faithfully propagate and nothing will ever correct. */
    var headingSeedTrusted: Boolean = false
        private set

    /** False when leveling had to complete on a window that never went quiet within
     * [PipelineConfig.maxLevelingWaitS]. Surface it: a session leveled this way likely
     * carries a horizontal-accel bias into Channel P for its entire duration, since
     * leveling happens once and is never redone. */
    var levelingTrusted: Boolean = true
        private set

    /** `WAITING_FOR_GNSS` covers two real states. Once a local frame exists, the
     * remaining prerequisite is a moving GNSS fix whose bearing can seed heading. */
    val waitingForMovingFix: Boolean
        get() = phase == PipelinePhase.WAITING_FOR_GNSS && localFrame != null

    private val forwardAxis = ForwardAxisEstimator()
    private val channelP = PhysicsSpeedChannel(config.physics)
    private val zuptDetector = ZuptDetector(config.zupt)

    private var ukf: DualChannelUkf? = null

    /** Constructed once [leveling] resolves (Stage12Channel needs a fixed mount
     * estimate to project accel/gyro into the same axis convention the model was
     * trained on - see that class's doc). Null for the whole session if
     * [stage12Model] was null, or forever if it was supplied but leveling never
     * completes (which only happens if the session ends first). */
    private var stage12Channel: Stage12Channel? = null

    /** Constructed eagerly (unlike [stage12Channel], which needs leveling first):
     * [CorridorChannel] has no leveling dependency, only a first GNSS fix to kick
     * off its road-network fetch. Null for the whole session if
     * [roadNetworkProvider] was null. */
    private val corridorChannel: CorridorChannel? = roadNetworkProvider?.let { CorridorChannel(it) }

    /** A second filter that receives no GNSS and no velocity channel during a
     * blackout: the honest "what the phone does today" CTCV coast. Having it on
     * screen next to the fused track is the A/B the demo is actually about. */
    private var coastUkf: DualChannelUkf? = null

    private val levelingWindow = ArrayList<DoubleArray>()
    private var levelingStartNs: Long? = null
    private var levelingWaitStartNs: Long? = null

    private var lastImuNs: Long? = null
    private var sessionStartNs: Long? = null

    private var pendingFix: GnssFix? = null
    private var lastFix: GnssFix? = null
    private var lastAdmittedSpeed: Double? = null
    private var lastSpeedDelta: Double = 0.0
    private var waitingForGnssSinceNs: Long? = null
    private var lastAdmittedFixNs: Long? = null
    private var cyclesSinceAxisEstimate: Int = 0

    var blackout: Boolean = false
        private set
    private var blackoutStartNs: Long? = null
    private var blackoutDistanceM: Double = 0.0
    private var blackoutTruthNorth: Double = 0.0
    private var blackoutTruthEast: Double = 0.0

    private var imuCount: Long = 0
    private var imuRateHz: Double = 0.0
    private var rateWindowStartNs: Long? = null
    private var rateWindowCount: Long = 0
    private var lastLatencyMs: Double = 0.0

    var lastSnapshot: FusionSnapshot? = null
        private set

    data class GnssFix(
        val tNs: Long,
        val latDeg: Double,
        val lonDeg: Double,
        val speedMps: Double,
        val bearingDeg: Double,
        val accuracyM: Double
    )

    /** Toggle whether GNSS reaches the filter. Fixes keep arriving and keep being
     * logged either way. */
    fun setBlackout(active: Boolean, tNs: Long) {
        if (active == blackout) return
        blackout = active
        if (active) {
            blackoutStartNs = tNs
            blackoutDistanceM = 0.0
            val fix = lastFix
            val frame = localFrame
            if (fix != null && frame != null) {
                val ne = frame.toNorthEast(fix.latDeg, fix.lonDeg)
                blackoutTruthNorth = ne[0]
                blackoutTruthEast = ne[1]
            }
            // Fork the coast baseline from the fused state at the moment the lights
            // go out, so the two tracks are comparable from a common starting point.
            forwardAxis.discardInterval()
            val current = ukf?.state()
            if (current != null) coastUkf = DualChannelUkf(current, config.fusion)
            // Freeze Stage 12's affine calibration on pre-blackout data only - see
            // Stage12Calibration's class doc for why this must not keep refitting
            // once blackout starts.
            stage12Channel?.onBlackoutStart()
            // Lock a candidate road at the outer UKF's position the instant the
            // blackout starts, seeded with whatever speed estimate is on hand -
            // matches canonical_evaluate_v4.py's CorridorFilter(road, s_init=0.0,
            // v_init=speeds[idx_start]).
            if (current != null) {
                corridorChannel?.onBlackoutStart(
                    pos = doubleArrayOf(current.posN, current.posE),
                    initialSpeed = current.speed
                )
            }
        } else {
            blackoutStartNs = null
            corridorChannel?.onBlackoutEnd()
        }
    }

    /**
     * Raw GNSS fix from LocationManager.GPS_PROVIDER. Never the fused provider: it
     * already blends IMU, WiFi and cell into its position, so measuring our dead
     * reckoning against it would be measuring Google's dead reckoning against ours.
     *
     * The fix is staged rather than applied immediately, so all filter work happens
     * on the IMU thread at a single well-defined cadence.
     */
    fun onGnss(fix: GnssFix) {
        lastFix = fix
        if (localFrame == null) localFrame = LocalFrame(fix.latDeg, fix.lonDeg)
        // Fires at most once per session - see CorridorChannel.onFirstFix's own
        // idempotency guard. Kicked off here rather than waiting for RUNNING phase
        // so the Overpass fetch has as much of the pre-blackout drive as possible
        // to complete before the first blackout needs a locked road.
        localFrame?.let { corridorChannel?.onFirstFix(fix.latDeg, fix.lonDeg, it) }

        if (blackout) {
            // Ground truth accounting continues through the blackout, using the
            // speed integral rather than summed fix-to-fix distance: independent
            // per-fix position noise inflates a summed path length, and a larger
            // denominator would make our own drift percentage smaller.
            val previous = lastAdmittedTruthNs
            if (previous != null) {
                val dt = (fix.tNs - previous) / NANOS_PER_SECOND
                if (dt in 0.0..5.0) blackoutDistanceM += fix.speedMps * dt
            }
            lastAdmittedTruthNs = fix.tNs
            return
        }

        lastAdmittedTruthNs = fix.tNs
        pendingFix = fix
    }

    private var lastAdmittedTruthNs: Long? = null

    /**
     * Raw IMU sample. `accel` is TYPE_ACCELEROMETER in m/s^2 with gravity included,
     * `gyro` is TYPE_GYROSCOPE in rad/s, both in the phone frame, both raw.
     *
     * `tNs` is SensorEvent.timestamp: nanoseconds on the monotonic elapsedRealtime
     * base, never wall clock. dt is computed per sample from consecutive timestamps,
     * because Android will not hand you a clean 10 ms cadence and pretending it does
     * injects error directly into the integration.
     */
    fun onImu(tNs: Long, accel: DoubleArray, gyro: DoubleArray): FusionSnapshot? {
        val startedNs = System.nanoTime()
        if (sessionStartNs == null) sessionStartNs = tNs
        imuCount++
        updateRate(tNs)

        when (phase) {
            PipelinePhase.LEVELING -> {
                if (levelingStartNs == null) levelingStartNs = tNs
                if (levelingWaitStartNs == null) levelingWaitStartNs = tNs

                // A sample above the quiet threshold means the window collected so
                // far includes handling motion, not just gravity: restart it here
                // rather than average that motion into the "up" vector.
                val gyroMagnitude = LinAlg.norm(gyro)
                if (gyroMagnitude > config.levelingGyroQuietThresholdRadS) {
                    levelingWindow.clear()
                    levelingStartNs = tNs
                }
                levelingWindow.add(accel.copyOf())

                val elapsed = (tNs - levelingStartNs!!) / NANOS_PER_SECOND
                val totalWaited = (tNs - levelingWaitStartNs!!) / NANOS_PER_SECOND
                val gaveUp = totalWaited >= config.maxLevelingWaitS

                if ((elapsed >= config.levelingWindowS || gaveUp) && levelingWindow.size >= 2) {
                    leveling = MountLeveling.fromStationaryWindow(levelingWindow)
                    levelingTrusted = elapsed >= config.levelingWindowS
                    levelingWindow.clear()
                    val model = stage12Model
                    if (model != null) stage12Channel = Stage12Channel(leveling!!, model, config.stage12)
                    phase = PipelinePhase.WAITING_FOR_GNSS
                }
                lastImuNs = tNs
                return null
            }

            PipelinePhase.WAITING_FOR_GNSS -> {
                val fix = pendingFix ?: lastFix
                val frame = localFrame
                if (fix == null || frame == null) {
                    lastImuNs = tNs
                    return null
                }

                // Do not initialise on a fix taken at a standstill.
                //
                // This filter has no heading measurement at all: psi is seeded once
                // and then propagated by the gyro, and the only thing that can
                // correct it afterwards is the covariance coupling from a velocity
                // residual. Seeding it while parked therefore sticks, and while
                // parked it is unseedable: GNSS bearing at a standstill is noise,
                // and the few metres of per-fix position noise push the filter's
                // velocity in random directions, which drags psi with it. Once the
                // vehicle moves off, Channel P rotates its speed by that wrong psi
                // and reinforces it.
                //
                // Measured before this wait existed, on the synthetic session with a
                // 1 Hz fix rate: 23 degrees of heading error by t=13 s and a fused
                // track hundreds of metres off with GNSS fully available. The Python
                // reference never hits this because its offline chain interpolates
                // GNSS onto the 100 Hz grid, so its filter gets a position update
                // every cycle and cannot wander between fixes. A phone gets one fix
                // a second. This is exactly the class of difference the front-end
                // gate exists to surface.
                val movingEnough = fix.speedMps >= config.minSpeedForBearingMps
                val waited = (tNs - (waitingForGnssSinceNs ?: tNs)) / NANOS_PER_SECOND
                if (waitingForGnssSinceNs == null) waitingForGnssSinceNs = tNs

                if (!movingEnough && waited < config.maxWaitForMovingFixS) {
                    lastImuNs = tNs
                    return null
                }
                headingSeedTrusted = movingEnough

                val ne = frame.toNorthEast(fix.latDeg, fix.lonDeg)
                val heading = bearingDegToPsi(fix.bearingDeg)
                val initial = UkfState(
                    posN = ne[0],
                    posE = ne[1],
                    velN = fix.speedMps * kotlin.math.cos(heading),
                    velE = fix.speedMps * kotlin.math.sin(heading),
                    heading = heading
                )
                ukf = DualChannelUkf(initial, config.fusion)
                channelP.reseed(fix.speedMps)
                pendingFix = null
                phase = PipelinePhase.RUNNING
                lastImuNs = tNs
                return null
            }

            PipelinePhase.RUNNING -> Unit
        }

        val previousNs = lastImuNs
        lastImuNs = tNs
        if (previousNs == null) return lastSnapshot

        var dt = (tNs - previousNs) / NANOS_PER_SECOND
        if (dt <= 0.0) return lastSnapshot
        if (dt > config.maxStepDtS) dt = config.maxStepDtS

        val level = leveling ?: return lastSnapshot
        val filter = ukf ?: return lastSnapshot

        val horizontal = level.horizontal(accel)
        val yawRate = level.yawRate(gyro)

        val fix = pendingFix
        pendingFix = null

        // Forward axis: accumulate at IMU rate while genuinely driving, with the sign
        // resolved against the most recent GNSS speed change. Accumulating only on
        // GNSS fixes would need 300 seconds of driving to reach the sample count the
        // offline version reaches in 3, which would leave Channel P silent through
        // most of a demo.
        val currentSpeed = fix?.speedMps ?: lastFix?.speedMps ?: 0.0
        // No samples are buffered during a blackout: there is no speed reference to
        // attribute them to, and attributing them to the next one that arrives is
        // what corrupts the axis.
        if (!blackout) forwardAxis.addSample(horizontal, currentSpeed, currentSpeed * yawRate)
        if (fix != null) {
            val previousSpeed = lastAdmittedSpeed
            lastSpeedDelta = if (previousSpeed == null) 0.0 else fix.speedMps - previousSpeed
            val intervalS = lastAdmittedFixNs?.let { (fix.tNs - it) / NANOS_PER_SECOND } ?: 1.0
            lastAdmittedSpeed = fix.speedMps
            lastAdmittedFixNs = fix.tNs
            // The delta covers the samples since the previous fix, so it closes that
            // interval rather than opening the next one.
            forwardAxis.closeInterval(lastSpeedDelta, intervalS)
        }
        cyclesSinceAxisEstimate++
        if (forwardAxis.hasEnough() && cyclesSinceAxisEstimate >= AXIS_RECOMPUTE_CYCLES) {
            forwardAxis.estimate()
            cyclesSinceAxisEstimate = 0
        }

        val zuptActive = config.zupt.enabled && zuptDetector.update(horizontal, yawRate)
        if (!config.zupt.enabled) zuptDetector.update(horizontal, yawRate)

        // Channel P. Until the forward axis is resolved the channel has nothing
        // trustworthy to integrate, so it reseeds on GNSS and otherwise says nothing
        // rather than integrating along an arbitrary direction.
        val axis = forwardAxis.axis
        var channelSpeed: Double? = null
        if (config.useChannelP) {
            if (zuptActive) {
                channelP.applyZupt()
            }
            channelSpeed = if (fix != null) {
                channelP.update(dt, 0.0, fix.speedMps)
            } else if (axis != null) {
                channelP.update(dt, horizontal[0] * axis[0] + horizontal[1] * axis[1])
            } else {
                null
            }
        }

        val gnssPos: DoubleArray?
        val gnssVel: DoubleArray?
        var gnssHeading: Double? = null
        val frame = localFrame
        if (fix != null && frame != null) {
            gnssPos = frame.toNorthEast(fix.latDeg, fix.lonDeg)
            val psi = bearingDegToPsi(fix.bearingDeg)
            gnssVel = doubleArrayOf(
                fix.speedMps * kotlin.math.cos(psi),
                fix.speedMps * kotlin.math.sin(psi)
            )
            // Course over ground is a real heading observation while moving and pure
            // noise at a standstill, so it is gated on speed here rather than inside
            // the filter.
            if (fix.speedMps >= config.minSpeedForBearingMps) gnssHeading = psi
        } else {
            gnssPos = null
            gnssVel = null
        }

        // Stage 12 keeps running (predicting, and while GNSS is available and it is
        // not a blackout, collecting calibration pairs) every cycle regardless of
        // blackout state - see Stage12Channel's class doc. Whether its output
        // actually reaches the filter below is decided here, not there: it only
        // takes Channel A's slot during a blackout, and only once it has a trusted
        // calibration; Channel P is the fallback otherwise, exactly as before this
        // channel existed.
        val stage12Output = stage12Channel?.update(
            tNs = tNs,
            accel = accel,
            gyro = gyro,
            leveledYawRate = yawRate,
            gnssSpeed = fix?.speedMps,
            blackout = blackout
        )
        val finalChannelASpeed = stage12Output?.calibratedSpeedMps ?: channelSpeed
        val finalRChannelA = when {
            stage12Output != null -> stage12Output.rChannelA
            channelSpeed != null -> config.physics.rMps
            else -> null
        }
        val finalGyroYaw = stage12Output?.correctedYawRateRadS ?: yawRate

        // Corridor (Mode F: Best Adaptive Hybrid). Same layered-gating shape as
        // Stage 12: the channel itself runs and tracks whenever a road is locked
        // on, but its road-snapped position only reaches the outer UKF during a
        // blackout, using the exact same speed/yaw evidence Channel A is already
        // getting this cycle - not a second, independent measurement. The
        // variance passed to the corridor filter's own update_speed mirrors
        // finalRChannelA's source but is never squared again the way
        // finalRChannelA is by the outer UKF's rChannelAOverride path (see
        // Stage12Output.rChannelA's doc): Channel P has no learned variance, so
        // its configured 1-sigma is squared here into a pseudo-variance instead.
        val corridorSpeedVariance = when {
            stage12Output != null -> stage12Output.rChannelA
            channelSpeed != null -> config.physics.rMps * config.physics.rMps
            else -> null
        }
        val priorState = filter.state()
        val corridorOutput = if (blackout && finalChannelASpeed != null && corridorSpeedVariance != null) {
            corridorChannel?.update(
                dt = dt,
                speedMeas = finalChannelASpeed,
                speedUncertainty = corridorSpeedVariance,
                yawRateRadS = finalGyroYaw,
                outerPositionNorthEast = doubleArrayOf(priorState.posN, priorState.posE)
            )
        } else {
            null
        }

        val state = filter.step(
            dt = dt,
            gyroYaw = finalGyroYaw,
            channelASpeed = finalChannelASpeed,
            channelBSpeed = null,
            gnssPos = gnssPos,
            gnssVel = gnssVel,
            rChannelAOverride = finalRChannelA,
            zupt = zuptActive,
            rZupt = config.zupt.rMps,
            gnssHeading = gnssHeading,
            roadSignaturePos = corridorOutput?.positionNorthEast,
            roadSignatureConfidence = corridorOutput?.confidence ?: 0.0
        )

        // Coast baseline: same gyro, no GNSS, no velocity channel, and no Stage 12
        // yaw correction either - this stays the honest "what the phone does today"
        // baseline the demo is about, unaffected by anything above.
        val coastState = coastUkf?.step(dt = dt, gyroYaw = yawRate) ?: state

        lastLatencyMs = (System.nanoTime() - startedNs) / 1_000_000.0

        val snapshot = buildSnapshot(
            tNs, state, coastState, channelSpeed,
            stage12SpeedMps = stage12Output?.calibratedSpeedMps,
            stage12Active = stage12Output != null,
            corridorRoadLocked = corridorChannel?.roadLocked == true,
            corridorConfidence = corridorOutput?.confidence ?: 0.0,
            corridorActive = corridorOutput != null &&
                corridorOutput.confidence >= config.fusion.roadSignatureConfidenceThreshold,
            corridorPositionCorrectionM = corridorOutput?.correctionMagnitudeM,
            zuptActive
        )
        lastSnapshot = snapshot
        return snapshot
    }

    private fun buildSnapshot(
        tNs: Long,
        state: UkfState,
        coast: UkfState,
        channelSpeed: Double?,
        stage12SpeedMps: Double?,
        stage12Active: Boolean,
        corridorRoadLocked: Boolean,
        corridorConfidence: Double,
        corridorActive: Boolean,
        corridorPositionCorrectionM: Double?,
        zuptActive: Boolean
    ): FusionSnapshot {
        val startNs = sessionStartNs ?: tNs
        val tSeconds = (tNs - startNs) / NANOS_PER_SECOND

        val frame = localFrame
        val fix = lastFix
        var truthNorth = 0.0
        var truthEast = 0.0
        if (frame != null && fix != null) {
            val ne = frame.toNorthEast(fix.latDeg, fix.lonDeg)
            truthNorth = ne[0]
            truthEast = ne[1]
        }

        // Live drift, against the withheld fixes. This is a display number. The
        // number that goes on a slide is computed offline by
        // tools/phone_replay/run.py from the raw log, so a bug in this arithmetic
        // cannot flatter the reported result.
        val driftMeters = if (blackout) {
            hypot(state.posN - truthNorth, state.posE - truthEast)
        } else {
            0.0
        }
        // Below this the metric is dividing by GPS noise, not distance: consumer GNSS
        // fix noise alone is 3-5 m, so at 1.1 m of blackout travel the ~4 m of that
        // noise reads as a 363% drift before the vehicle has gone anywhere. Waiting
        // for 15 m of blackout travel keeps the denominator meaningfully larger than
        // the fix noise it is being divided against.
        val driftPercent = if (blackout && blackoutDistanceM > 15.0) {
            driftMeters / blackoutDistanceM * 100.0
        } else {
            0.0
        }

        val blackoutElapsed = blackoutStartNs?.let { (tNs - it) / NANOS_PER_SECOND } ?: 0.0

        return FusionSnapshot(
            tSeconds = tSeconds,
            fusedNorth = state.posN,
            fusedEast = state.posE,
            coastNorth = coast.posN,
            coastEast = coast.posE,
            speedMps = state.speed,
            headingDeg = psiToBearingDeg(state.heading),
            blackout = blackout,
            blackoutElapsedS = blackoutElapsed,
            blackoutDistanceM = blackoutDistanceM,
            driftMeters = driftMeters,
            driftPercent = driftPercent,
            channelPSpeed = channelSpeed,
            stage12SpeedMps = stage12SpeedMps,
            stage12Active = stage12Active,
            corridorRoadLocked = corridorRoadLocked,
            corridorConfidence = corridorConfidence,
            corridorActive = corridorActive,
            corridorPositionCorrectionM = corridorPositionCorrectionM,
            zuptActive = zuptActive,
            imuRateHz = imuRateHz,
            stepLatencyMs = lastLatencyMs,
            phase = phase,
            lastTruthNorth = truthNorth,
            lastTruthEast = truthEast
        )
    }

    private fun updateRate(tNs: Long) {
        val windowStart = rateWindowStartNs
        if (windowStart == null) {
            rateWindowStartNs = tNs
            rateWindowCount = 0
            return
        }
        rateWindowCount++
        val elapsed = (tNs - windowStart) / NANOS_PER_SECOND
        if (elapsed >= 1.0) {
            imuRateHz = rateWindowCount / elapsed
            rateWindowStartNs = tNs
            rateWindowCount = 0
        }
    }

    /** Diagnostics for the UI and for a recording's own header. */
    fun describeFrontEnd(): String {
        val level = leveling
        val axis = forwardAxis.axis
        val gravity = level?.gravityMagnitude ?: 0.0
        val axisText = if (axis == null) {
            "not resolved (${forwardAxis.sampleCount} driving samples)"
        } else {
            "[${fmt(axis[0])}, ${fmt(axis[1])}]" +
                if (forwardAxis.signResolved) "" else " SIGN NOT RESOLVED"
        }
        return "gravity ${fmt(gravity)} m/s^2, forward axis $axisText, imu ${fmt(imuRateHz)} Hz"
    }

    private fun fmt(value: Double): String {
        val scaled = kotlin.math.round(value * 1000.0) / 1000.0
        return if (abs(scaled) < 1e-9) "0.0" else scaled.toString()
    }

    /** Total IMU samples seen, for the log header and the sanity check that the
     * foreground service was never throttled. */
    val imuSamples: Long get() = imuCount

    val zuptStatistics: ZuptStatistics get() = zuptDetector.lastStatistics

    val forwardAxisResolved: Boolean get() = forwardAxis.axis != null

    /** The resolved forward axis in the level frame, or null while it is still being
     * estimated. Exposed so the offline harness can compare it against the Python
     * reference's axis component by component rather than trusting a boolean. */
    val forwardAxisValue: DoubleArray? get() = forwardAxis.axis

    val channelPSpeedMps: Double? get() = channelP.speed

    val blackoutDistanceMeters: Double get() = max(0.0, blackoutDistanceM)
}
