package org.sih26.deadreckoning.replay

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

    fun timelineCountUpTo(t: Double): Int = timeline.lastIndexAtOrBefore(t) { it.t } + 1

    fun truthAt(t: Double): TruthFix? = truth.getOrNull(truth.lastIndexAtOrBefore(t) { it.t })
    fun fusedAt(t: Double): ReplayPoint? = fused.getOrNull(fused.lastIndexAtOrBefore(t) { it.t })
    fun coastAt(t: Double): ReplayPoint? = coast.getOrNull(coast.lastIndexAtOrBefore(t) { it.t })
    fun stage12At(t: Double): ReplayPoint? = stage12.getOrNull(stage12.lastIndexAtOrBefore(t) { it.t })
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
