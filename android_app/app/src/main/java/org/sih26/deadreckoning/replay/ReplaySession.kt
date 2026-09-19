package org.sih26.deadreckoning.replay

import kotlin.math.hypot

/**
 * What a replay plays back, independent of where it came from. Deliberately free of
 * Android and Compose imports, same reasoning as the fusion package: the parsing in
 * [ReplayLoader] can then be exercised on a desktop JVM against a recorded file.
 *
 * Everything is in the session's local tangent frame (metres, north/east from the
 * first GNSS fix, [org.sih26.deadreckoning.fusion.LocalFrame]) and on the session's own
 * clock in seconds, so tracks from different streams line up without conversion.
 */
enum class ReplaySource { PHONE_SESSION, IOVNBD_CSV }

enum class ReplayTrack { TRUTH, FUSED, COAST, STAGE12 }

data class ReplayPoint(val t: Double, val north: Double, val east: Double, val track: ReplayTrack)

/** One GNSS fix. Optional fields are null when the source does not carry them
 * (IO-VNBD has satellite counts but no withheld flag; a phone session is the reverse). */
data class TruthFix(
    val t: Double,
    val north: Double,
    val east: Double,
    val speedMps: Double?,
    val accuracyM: Double?,
    val headingDeg: Double?,
    val satellites: Int?,
    /** True when the recording app withheld this fix from its live filter (a software
     * blackout). The fix itself is still real GNSS truth and is plotted as such. */
    val withheld: Boolean
)

/**
 * One simulated GNSS outage on an imported drive: between [startS] and [endS] the
 * Stage 12 trajectory was denied every GNSS fix and had to dead-reckon on the model's
 * own speed and yaw output. The real fixes are still in the file (and still plotted as
 * truth), which is what makes the divergence measurable after the fact.
 *
 * The three metrics are worked out once, on import, against the real GNSS track
 * (linearly interpolated between its ~1 Hz fixes); null when the outage produced no
 * dead-reckoned point to measure.
 */
data class BlackoutWindow(
    val startS: Double,
    val endS: Double,
    /** Distance between Stage 12 and real GNSS at the last point before GNSS returned. */
    val endErrorM: Double?,
    /** Worst distance between Stage 12 and real GNSS at any point of the outage. */
    val maxErrorM: Double?,
    /** How far the real GNSS track travelled during the outage - the denominator of drift. */
    val gnssDistanceM: Double?
) {
    val durationS: Double get() = endS - startS

    /** End error as a percentage of distance travelled, the SIH scorecard's own
     * definition of drift. Null when the vehicle barely moved (a percentage of ~0 m is
     * noise, not a metric). */
    val driftPercent: Double?
        get() = if (endErrorM != null && gnssDistanceM != null && gnssDistanceM > 5.0) endErrorM / gnssDistanceM * 100.0 else null
}

class ReplaySession(
    val label: String,
    val source: ReplaySource,
    val startS: Double,
    val endS: Double,
    val truth: List<TruthFix>,
    /** The recording app's own live fused / coast output, decimated. Empty for sources
     * that never ran the filter (IO-VNBD). */
    val fused: List<ReplayPoint>,
    val coast: List<ReplayPoint>,
    /** A second trajectory built on import, not read from the log: the Stage 12
     * model's own calibrated speed + corrected yaw rate, dead-reckoned from the raw
     * IMU stream for the whole drive rather than only during a real blackout - see
     * [ReplayLoader]'s buildStage12Trajectory. Empty when the source carries no raw
     * IMU (IO-VNBD) or no Stage 12 model was available to the loader. */
    val stage12: List<ReplayPoint> = emptyList(),
    /** The simulated GNSS outages [stage12] was run through, in time order. While none
     * is active [stage12] is simply the GNSS track, so it sits on top of truth; inside
     * one it is model-only dead reckoning and free to diverge. Empty whenever [stage12]
     * is, and for a phone session (whose outages, if any, are the real ones flagged by
     * [TruthFix.withheld]). */
    val blackouts: List<BlackoutWindow> = emptyList(),
    /** The session's [org.sih26.deadreckoning.fusion.LocalFrame] origin fix, so the
     * Replay screen's map can project these North/East tracks onto real lat/lon.
     * Null only if the file had no usable fix at all, which [ReplayLoader] already
     * refuses to return a session for. */
    val originLat0Deg: Double? = null,
    val originLon0Deg: Double? = null
) {
    /** All tracks merged in time order, so "everything up to time t" is a prefix
     * ([timelineCountUpTo]) instead of a per-frame filter. */
    val timeline: List<ReplayPoint> = (
        truth.map { ReplayPoint(it.t, it.north, it.east, ReplayTrack.TRUTH) } + fused + coast + stage12
        ).sortedBy { it.t }

    val hasFusion: Boolean get() = fused.isNotEmpty() || coast.isNotEmpty()
    val hasStage12: Boolean get() = stage12.isNotEmpty()
    val hasSimulatedBlackouts: Boolean get() = blackouts.isNotEmpty()

    /** The simulated outage that contains [t], if any. */
    fun blackoutAt(t: Double): BlackoutWindow? = blackouts.firstOrNull { t >= it.startS && t < it.endS }

    /** How far the Stage 12 track is from real GNSS at the moment of its latest point.
     * Compared at the Stage 12 point's own time, against GNSS interpolated to that same
     * instant, so that a stale ~1 Hz fix is not mistaken for divergence: while GNSS is
     * feeding Stage 12 this reads ~0, and inside an outage it is the true drift. Null
     * if there is no Stage 12 point yet or it is over a second old. */
    fun stage12GapAt(t: Double): Double? {
        val p = stage12At(t) ?: return null
        if (t - p.t > 1.0) return null
        val truthPos = interpolateTruth(truth, p.t) ?: return null
        return hypot(p.north - truthPos[0], p.east - truthPos[1])
    }

    fun timelineCountUpTo(t: Double): Int = timeline.lastIndexAtOrBefore(t) { it.t } + 1

    fun truthAt(t: Double): TruthFix? = truth.getOrNull(truth.lastIndexAtOrBefore(t) { it.t })
    fun fusedAt(t: Double): ReplayPoint? = fused.getOrNull(fused.lastIndexAtOrBefore(t) { it.t })
    fun coastAt(t: Double): ReplayPoint? = coast.getOrNull(coast.lastIndexAtOrBefore(t) { it.t })
    fun stage12At(t: Double): ReplayPoint? = stage12.getOrNull(stage12.lastIndexAtOrBefore(t) { it.t })
}

/** Real GNSS position at time [t], linearly interpolated between the two surrounding
 * fixes (clamped to the first/last fix outside the track). [north, east] in metres, or
 * null for an empty track. */
internal fun interpolateTruth(truth: List<TruthFix>, t: Double): DoubleArray? {
    if (truth.isEmpty()) return null
    val i = truth.lastIndexAtOrBefore(t) { it.t }
    if (i < 0) return doubleArrayOf(truth[0].north, truth[0].east)
    if (i >= truth.size - 1) return doubleArrayOf(truth[i].north, truth[i].east)
    val a = truth[i]
    val b = truth[i + 1]
    val span = b.t - a.t
    val f = if (span <= 1e-6) 0.0 else ((t - a.t) / span).coerceIn(0.0, 1.0)
    return doubleArrayOf(a.north + f * (b.north - a.north), a.east + f * (b.east - a.east))
}

/** Index of the last element whose time is <= [t] in a time-sorted list, or -1. */
internal inline fun <T> List<T>.lastIndexAtOrBefore(t: Double, time: (T) -> Double): Int {
    var lo = 0
    var hi = size - 1
    var answer = -1
    while (lo <= hi) {
        val mid = (lo + hi) ushr 1
        if (time(this[mid]) <= t) {
            answer = mid
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    return answer
}
