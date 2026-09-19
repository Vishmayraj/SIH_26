package org.sih26.deadreckoning.fusion

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sin

/**
 * Kotlin port of fusion_core/python_prototype/ukf.py, MIP Section 5.
 *
 * State, process model, update order, trust weighting and the GNSS re-admission ramp
 * are all described there and are not re-argued here. What this file owes the reader
 * is the things a port can get wrong, so they are commented where they happen.
 *
 * The Python reference wraps filterpy. This port reimplements the filterpy pieces it
 * uses, because there is no filterpy on Android, and it reimplements them
 * deliberately literally: same sigma-point convention, same weights, same sequential
 * update order, same covariance repair points. Anything "improved" in passing would
 * break the parity fixture, and the fixture is worth more than the improvement.
 *
 * Verified by tools/parity/run_kotlin_parity.sh, which replays the committed
 * 600-cycle fixture through this class on the JVM and fails the build on divergence.
 */

const val N_STATES = 7

const val PN = 0
const val PE = 1
const val VN = 2
const val VE = 3
const val PSI = 4
const val BA = 5
const val BG = 6

/** Smallest eigenvalue symmetrizeP will let P hold. Matches the Python reference. */
private const val MIN_EIGENVALUE = 1e-9

data class FusionConfig(
    // Sigma-point scaling. alpha=1.0 is not the MIP's suggested 1e-3: see the long
    // comment on FusionConfig.alpha in the Python reference. Short version, because
    // it is the single most load-bearing constant in this file: with alpha=1e-3 the
    // zeroth covariance weight is about -1e6 while the sigma points are squeezed to
    // a fraction of a standard deviation, so floating-point cancellation in fx gets
    // multiplied by a million every cycle. It survived only because velocity-channel
    // updates kept dragging P back down. The phone runs the configuration without
    // those updates, which is the one that reaches diag(P) around 1e24 and goes
    // indefinite at GNSS reacquisition. Do not "restore" this to the MIP value.
    val alpha: Double = 1.0,
    val beta: Double = 2.0,
    val kappa: Double = 0.0,

    val qPosGnss: Double = 0.05,
    val qVelGnss: Double = 0.1,
    val qPosBlackout: Double = 0.5,
    val qVelBlackout: Double = 0.2,
    val qPsi: Double = 1e-3,
    val qBa: Double = 1e-5,
    val qBg: Double = 1e-6,

    val rGnssPos: Double = 3.0,
    val rGnssVel: Double = 0.3,
    /** When true, zero the cross-covariance between the angular states (psi, ba, bg)
     * and (vn, ve) immediately before a Channel A/B update. Ported from the Python
     * reference (ukf.py's `covariance_decoupling`, default true there too): a scalar
     * speed update has no way to say which direction that speed points, but without
     * this it can still drag heading and bias around through whatever cross-
     * covariance the previous cycle's sigma points happened to build up between
     * speed and psi. That coupling is real information when a *directional*
     * velocity measurement (old rotated-into-N/E Channel A/B, or GNSS velocity)
     * supplied it; it is spurious windup once Channel A/B became the scalar
     * `hxSpeed` measurement below, which asserts nothing about direction at all. */
    val covarianceDecoupling: Boolean = true,
    val rChannelA: Double = 0.5,
    val rChannelB: Double = 1.75,
    /** GNSS course over ground, 1-sigma in radians. About 3 degrees, the right order
     * for a consumer GNSS course at road speed. See hxHeading for why this source
     * exists. Only applied when the caller supplies a heading, and the caller gates
     * that on speed: course at a standstill is noise, and feeding noise to the one
     * thing that observes psi is worse than not observing psi at all. */
    val rGnssHeading: Double = 0.05,

    val channelAbDisagreementFactor: Double = 1.5,
    val channelAInflatedRMultiplier: Double = 5.0,

    val roadSignatureConfidenceThreshold: Double = 0.85,

    // Off by default, and measured net-neutral-to-negative whenever a velocity
    // channel is already active, because rotating a scalar speed by the heading
    // estimate already asserts zero lateral velocity. Exposed as a runtime toggle
    // rather than a compile-time default so the A/B can be shown on real roads.
    val enableNhc: Boolean = false,
    val rNhc: Double = 0.3,

    val gnssReacquireRampS: Double = 2.5,
    val gnssReacquireRMultiplierStart: Double = 50.0,

    val initialPDiag: DoubleArray = doubleArrayOf(1.0, 1.0, 0.5, 0.5, 0.05, 0.01, 1e-4)
)

data class UkfState(
    val posN: Double,
    val posE: Double,
    val velN: Double,
    val velE: Double,
    val heading: Double
) {
    val speed: Double get() = hypot(velN, velE)
}

/** Process model: constant-velocity, constant-turn-rate. Speed magnitude is held
 * across the step and only rotated into the new heading; velocity changes arrive
 * through measurement updates, never by integrating accelerometer here. */
fun fx(x: DoubleArray, dt: Double, gyroYaw: Double): DoubleArray {
    val out = x.copyOf()
    val yawRate = gyroYaw - x[BG]
    val psiNew = x[PSI] + yawRate * dt

    val speed = hypot(x[VN], x[VE])
    val vnNew = speed * cos(psiNew)
    val veNew = speed * sin(psiNew)

    out[PN] = x[PN] + (x[VN] + vnNew) / 2.0 * dt
    out[PE] = x[PE] + (x[VE] + veNew) / 2.0 * dt
    out[VN] = vnNew
    out[VE] = veNew
    out[PSI] = psiNew
    return out
}

private fun hxPosition(x: DoubleArray) = doubleArrayOf(x[PN], x[PE])

private fun hxPositionVelocity(x: DoubleArray) = doubleArrayOf(x[PN], x[PE], x[VN], x[VE])

private fun hxVelocity(x: DoubleArray) = doubleArrayOf(x[VN], x[VE])

/** Scalar speed measurement, independent of heading direction. Matches the Python
 * reference's `hx_speed`. This is what Channel A/B are measured against now, instead
 * of the older approach of rotating the scalar speed into (vn, ve) using the filter's
 * own heading estimate and treating that as a 2-D velocity observation - the old
 * approach quietly asserted a direction the channel never actually measured. */
private fun hxSpeed(x: DoubleArray) = doubleArrayOf(hypot(x[VN], x[VE]))

/**
 * Direct heading measurement, for GNSS course over ground.
 *
 * Why this exists when MIP Section 5.3 has no such source: fx recomputes velocity
 * every cycle as speed * [cos(psi), sin(psi)], so velocity direction is a function of
 * psi rather than an independent quantity. A measurement that corrects (vn, ve)
 * without also moving psi is undone by the very next prediction step, and nothing
 * else in the filter observes psi, so psi is seeded once and then propagated
 * open-loop by the gyro forever.
 *
 * The offline chain hides this: it interpolates 1 Hz GNSS onto a 100 Hz grid, so its
 * filter gets a hundred position updates a second and heading is constrained
 * implicitly. A phone gets one fix a second. Measured on the synthetic session
 * replayed at a true 1 Hz fix rate, without this source, the fused track ended 59 m
 * off with GNSS available throughout.
 */
private fun hxHeading(x: DoubleArray) = doubleArrayOf(x[PSI])

/** Non-holonomic constraint: body-frame lateral velocity should be near zero. */
private fun hxNhc(x: DoubleArray) =
    doubleArrayOf(-x[VN] * sin(x[PSI]) + x[VE] * cos(x[PSI]))

/**
 * Van der Merwe scaled sigma points, matching filterpy's MerweScaledSigmaPoints.
 *
 * The convention that matters: filterpy factors (n + lambda) * P with scipy's
 * cholesky, which returns the upper factor U where P' = U^T U, and then adds and
 * subtracts the ROWS of U. Row k of U equals column k of the lower factor L, so this
 * reads columns of L. That single transpose is the easiest silent divergence in the
 * whole port.
 */
class MerweSigmaPoints(
    private val n: Int,
    alpha: Double,
    beta: Double,
    kappa: Double
) {
    private val lambda: Double = alpha * alpha * (n + kappa) - n
    val wm: DoubleArray
    val wc: DoubleArray
    val count: Int get() = 2 * n + 1

    init {
        val c = 0.5 / (n + lambda)
        wm = DoubleArray(2 * n + 1) { c }
        wc = DoubleArray(2 * n + 1) { c }
        wm[0] = lambda / (n + lambda)
        wc[0] = lambda / (n + lambda) + (1.0 - alpha * alpha + beta)
    }

    fun sigmaPoints(x: DoubleArray, p: Array<DoubleArray>): Array<DoubleArray> {
        val scaled = LinAlg.scale(p, n + lambda)
        val l = LinAlg.choleskyLower(scaled)

        val sigmas = Array(2 * n + 1) { DoubleArray(n) }
        sigmas[0] = x.copyOf()
        for (k in 0 until n) {
            for (i in 0 until n) {
                // Column k of L, which is row k of the upper factor filterpy uses.
                val offset = l[i][k]
                sigmas[k + 1][i] = x[i] + offset
                sigmas[n + k + 1][i] = x[i] - offset
            }
        }
        return sigmas
    }
}

private data class Transformed(val mean: DoubleArray, val cov: Array<DoubleArray>)

private fun unscentedTransform(
    sigmas: Array<DoubleArray>,
    wm: DoubleArray,
    wc: DoubleArray,
    noiseCov: Array<DoubleArray>?
): Transformed {
    val dim = sigmas[0].size
    val mean = DoubleArray(dim)
    for (i in sigmas.indices) {
        for (j in 0 until dim) mean[j] += wm[i] * sigmas[i][j]
    }

    val cov = LinAlg.zeros(dim, dim)
    for (i in sigmas.indices) {
        val y = DoubleArray(dim) { sigmas[i][it] - mean[it] }
        for (r in 0 until dim) {
            val wcy = wc[i] * y[r]
            for (c in 0 until dim) cov[r][c] += wcy * y[c]
        }
    }
    if (noiseCov != null) {
        for (r in 0 until dim) for (c in 0 until dim) cov[r][c] += noiseCov[r][c]
    }
    return Transformed(mean, cov)
}

/**
 * The filter. One instance per session; call [step] once per cycle.
 *
 * Thread confinement: this class is not synchronised. It is owned by the fusion
 * thread and nothing else touches it, which is cheaper and easier to reason about
 * than locking a filter on a 100 Hz hot path.
 */
class DualChannelUkf(initialState: UkfState, val config: FusionConfig = FusionConfig()) {

    private val points = MerweSigmaPoints(N_STATES, config.alpha, config.beta, config.kappa)

    var x: DoubleArray = DoubleArray(N_STATES)
        private set
    var p: Array<DoubleArray> = LinAlg.diag(config.initialPDiag)
        private set

    private var sigmasF: Array<DoubleArray> = Array(points.count) { DoubleArray(N_STATES) }

    private var timeSinceGnssReacquiredS: Double? = null
    private var wasBlackedOut: Boolean = false

    init {
        x[PN] = initialState.posN
        x[PE] = initialState.posE
        x[VN] = initialState.velN
        x[VE] = initialState.velE
        x[PSI] = initialState.heading
    }

    fun state(): UkfState = UkfState(x[PN], x[PE], x[VN], x[VE], x[PSI])

    private fun processNoise(dt: Double, blackout: Boolean): Array<DoubleArray> {
        val qPos = if (blackout) config.qPosBlackout else config.qPosGnss
        val qVel = if (blackout) config.qVelBlackout else config.qVelGnss
        return LinAlg.diag(
            doubleArrayOf(qPos, qPos, qVel, qVel, config.qPsi, config.qBa, config.qBg)
                .map { it * dt }
                .toDoubleArray()
        )
    }

    private fun channelAR(channelASpeed: Double, channelBSpeed: Double?): Double {
        // Section 5.4 is a disagreement detector. With one channel there is nothing
        // to disagree with, so A keeps its nominal R rather than being inflated or
        // deflated on no evidence.
        if (channelBSpeed == null) return config.rChannelA
        val disagreement = abs(channelASpeed - channelBSpeed)
        return if (disagreement > config.channelAbDisagreementFactor * config.rChannelB) {
            config.rChannelA * config.channelAInflatedRMultiplier
        } else {
            config.rChannelA
        }
    }

    private fun gnssRMultiplier(dt: Double, gnssAvailable: Boolean): Double {
        if (!gnssAvailable) {
            wasBlackedOut = true
            timeSinceGnssReacquiredS = null
            return 1.0
        }

        if (wasBlackedOut) {
            val elapsed = timeSinceGnssReacquiredS
            timeSinceGnssReacquiredS = if (elapsed == null) 0.0 else elapsed + dt

            val rampFrac = min(1.0, timeSinceGnssReacquiredS!! / config.gnssReacquireRampS)
            if (rampFrac >= 1.0) {
                wasBlackedOut = false
                return 1.0
            }
            val start = config.gnssReacquireRMultiplierStart
            return start + (1.0 - start) * rampFrac
        }
        return 1.0
    }

    /** Symmetrize P and floor its eigenvalues. Repeated sequential updates make P
     * drift asymmetric, and a state variance can collapse toward zero until Cholesky
     * sees a tiny negative. Flooring the eigenvalues guarantees positive
     * definiteness where adding jitter to the diagonal only sometimes does. This is
     * insurance, not the cure for the alpha problem described in FusionConfig. */
    private fun symmetrizeP() {
        val symmetric = LinAlg.symmetrize(p)
        val (values, vectors) = LinAlg.jacobiEigenSymmetric(symmetric)

        var floored = false
        val flooredValues = DoubleArray(values.size) { i ->
            if (values[i] < MIN_EIGENVALUE) {
                floored = true
                MIN_EIGENVALUE
            } else {
                values[i]
            }
        }

        if (!floored) {
            p = symmetric
            return
        }
        p = LinAlg.matMul(
            LinAlg.matMul(vectors, LinAlg.diag(flooredValues)),
            LinAlg.transpose(vectors)
        )
    }

    /** Zero the cross-covariance between the angular states (psi, ba, bg) and the
     * Cartesian velocity states (vn, ve). See [FusionConfig.covarianceDecoupling]. */
    private fun decoupleVelocityFromAngular() {
        for (angIdx in intArrayOf(PSI, BA, BG)) {
            for (velIdx in intArrayOf(VN, VE)) {
                p[angIdx][velIdx] = 0.0
                p[velIdx][angIdx] = 0.0
            }
        }
    }

    /** Regenerate sigma points from the current (just updated) x and P.
     *
     * filterpy reuses the sigma points produced by predict() for every subsequent
     * update in the same cycle, because its normal case is one predict and one
     * update. This filter does up to five sequential updates per cycle, so without
     * this each later update would fold its correction into the pre-update spread.
     * That is mathematically inconsistent and drives P non-positive-definite within
     * a handful of cycles. Call between every pair of sequential updates, and not
     * right after predict, which already leaves correct points in place. */
    private fun refreshSigmas() {
        sigmasF = points.sigmaPoints(x, p)
    }

    private fun predict(dt: Double, gyroYaw: Double, q: Array<DoubleArray>) {
        val sigmas = points.sigmaPoints(x, p)
        sigmasF = Array(sigmas.size) { fx(sigmas[it], dt, gyroYaw) }
        val transformed = unscentedTransform(sigmasF, points.wm, points.wc, q)
        x = transformed.mean
        p = transformed.cov
    }

    private fun update(
        z: DoubleArray,
        r: Array<DoubleArray>,
        hx: (DoubleArray) -> DoubleArray
    ) {
        val sigmasH = Array(sigmasF.size) { hx(sigmasF[it]) }
        val transformed = unscentedTransform(sigmasH, points.wm, points.wc, r)
        val zp = transformed.mean
        val s = transformed.cov
        val si = LinAlg.inverse(s)

        // Cross covariance uses the current (pre-update) x, matching filterpy.
        val pxz = LinAlg.zeros(N_STATES, z.size)
        for (i in sigmasF.indices) {
            val dx = DoubleArray(N_STATES) { sigmasF[i][it] - x[it] }
            val dz = DoubleArray(z.size) { sigmasH[i][it] - zp[it] }
            for (r0 in 0 until N_STATES) {
                val w = points.wc[i] * dx[r0]
                for (c in z.indices) pxz[r0][c] += w * dz[c]
            }
        }

        val k = LinAlg.matMul(pxz, si)
        val y = LinAlg.subVec(z, zp)
        x = LinAlg.addVec(x, LinAlg.matVec(k, y))
        p = LinAlg.subtract(p, LinAlg.matMul(LinAlg.matMul(k, s), LinAlg.transpose(k)))
    }

    /**
     * Advance one cycle.
     *
     * Null GNSS arguments mean a blackout cycle; null channel speeds mean that
     * channel has nothing to say and its update is skipped entirely. Both channels
     * null is the honest "no trained velocity channel" baseline, where the filter
     * coasts on the process model alone.
     */
    fun step(
        dt: Double,
        gyroYaw: Double,
        channelASpeed: Double? = null,
        channelBSpeed: Double? = null,
        gnssPos: DoubleArray? = null,
        gnssVel: DoubleArray? = null,
        roadSignaturePos: DoubleArray? = null,
        roadSignatureConfidence: Double = 0.0,
        rChannelAOverride: Double? = null,
        zupt: Boolean = false,
        rZupt: Double = 0.05,
        gnssHeading: Double? = null
    ): UkfState {
        val blackout = gnssPos == null

        predict(dt, gyroYaw, processNoise(dt, blackout))
        symmetrizeP()

        if (gnssPos != null) {
            val rMult = gnssRMultiplier(dt, gnssAvailable = true)
            if (gnssVel != null) {
                val z = doubleArrayOf(gnssPos[0], gnssPos[1], gnssVel[0], gnssVel[1])
                val r = LinAlg.diag(
                    doubleArrayOf(
                        config.rGnssPos * config.rGnssPos,
                        config.rGnssPos * config.rGnssPos,
                        config.rGnssVel * config.rGnssVel,
                        config.rGnssVel * config.rGnssVel
                    ).map { it * rMult * rMult }.toDoubleArray()
                )
                update(z, r, ::hxPositionVelocity)
            } else {
                val variance = config.rGnssPos * config.rGnssPos * rMult * rMult
                update(gnssPos.copyOf(), LinAlg.diag(doubleArrayOf(variance, variance)), ::hxPosition)
            }
            symmetrizeP()
            refreshSigmas()

            // GNSS course over ground as a direct heading measurement. The
            // measurement is pre-unwrapped against the current estimate rather than
            // installing a custom residual function, so the plain-subtraction
            // residual used everywhere else stays correct here too, including when
            // psi has run unwrapped past several full turns.
            if (gnssHeading != null) {
                val psiNow = x[PSI]
                val z = doubleArrayOf(psiNow + wrapToPi(gnssHeading - psiNow))
                update(z, LinAlg.diag(doubleArrayOf(config.rGnssHeading * config.rGnssHeading)), ::hxHeading)
                symmetrizeP()
                refreshSigmas()
            }
        } else {
            gnssRMultiplier(dt, gnssAvailable = false)
        }

        // Channel A and Channel B are scalar forward-speed measurements (hxSpeed),
        // independent of heading direction - see hxSpeed and covarianceDecoupling.
        if (channelASpeed != null) {
            val rA = rChannelAOverride ?: channelAR(channelASpeed, channelBSpeed)
            if (config.covarianceDecoupling) decoupleVelocityFromAngular()
            update(doubleArrayOf(channelASpeed), LinAlg.diag(doubleArrayOf(rA * rA)), ::hxSpeed)
            symmetrizeP()
            refreshSigmas()
        }

        if (channelBSpeed != null) {
            val rB = config.rChannelB
            if (config.covarianceDecoupling) decoupleVelocityFromAngular()
            update(doubleArrayOf(channelBSpeed), LinAlg.diag(doubleArrayOf(rB * rB)), ::hxSpeed)
            symmetrizeP()
        }

        // After the velocity channels, so a detected stop overrides whatever they
        // claimed. A drifting integrated-speed channel is the thing ZUPT is here to
        // correct, so letting it have the last word would defeat the purpose.
        if (zupt) {
            refreshSigmas()
            update(doubleArrayOf(0.0, 0.0), LinAlg.diag(doubleArrayOf(rZupt * rZupt, rZupt * rZupt)), ::hxVelocity)
            symmetrizeP()
        }

        if (config.enableNhc) {
            refreshSigmas()
            update(doubleArrayOf(0.0), LinAlg.diag(doubleArrayOf(config.rNhc * config.rNhc)), ::hxNhc)
            symmetrizeP()
        }

        if (roadSignaturePos != null &&
            roadSignatureConfidence >= config.roadSignatureConfidenceThreshold
        ) {
            refreshSigmas()
            val variance = (config.rGnssPos * 2.0) * (config.rGnssPos * 2.0)
            update(roadSignaturePos.copyOf(), LinAlg.diag(doubleArrayOf(variance, variance)), ::hxPosition)
            symmetrizeP()
        }

        return state()
    }
}
