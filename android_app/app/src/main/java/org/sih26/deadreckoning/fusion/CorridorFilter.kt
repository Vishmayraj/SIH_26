package org.sih26.deadreckoning.fusion

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Kotlin port of tools/stage12/corridor_filter.py: `CorridorRoad`, `CorridorFilter`,
 * `RoadProgressMatcher` - the 1-D along-road Kalman filter behind
 * `canonical_evaluate_v4.py`'s "Mode F: Best Adaptive Hybrid (Corridor + Anchors)",
 * the best-performing configuration in that scorecard.
 *
 * Android-free, same as the rest of this package: [RoadNetwork] here is a plain data
 * holder (list of [RoadSegment]), not the OSM-fetching client. The actual Overpass
 * query lives in the sensors package's `OverpassRoadNetworkProvider`, behind
 * [RoadNetworkProvider] so this file (and its logic) is testable on a desktop JVM
 * against a hand-built segment list, same reasoning as Stage12Channel's split from
 * MotionSpeedNetOnnx.
 *
 * Coordinate/heading convention throughout: North/East metres from [LocalFrame],
 * heading (psi) measured from north, positive toward east - `atan2(east, north)`,
 * matching Ukf.kt's fx and Geo.kt's psiToBearingDeg. `map_provider.py`'s own inline
 * comment worked this convention out by cross-referencing its evaluate.py caller;
 * reproduced here directly rather than re-derived.
 */

/** One OSM way segment (a single edge between two consecutive geometry points),
 * projected into the session's local North/East frame. No edge/node ID: the
 * classes this file ports (`CorridorRoad`'s chaining, `RoadNetwork`'s nearby-
 * segment search) key off physical proximity only, matching corridor_filter.py /
 * map_provider.py exactly - OSM node continuity (`matcher.py`'s `RealisticMatcher`,
 * the edge_id continuity penalty) is a different, HMM-style matcher that
 * `canonical_evaluate_v4.py`'s corridor/hybrid evaluator never calls, so it is not
 * bridged here either. */
data class RoadSegment(
    val p1: DoubleArray,
    val p2: DoubleArray
) {
    val length: Double = hypot(p2[0] - p1[0], p2[1] - p1[1])

    /** atan2(east, north) - see the class doc's convention note. */
    val heading: Double = atan2(p2[1] - p1[1], p2[0] - p1[0])
}

/** Plain segment list, exactly `map_provider.py`'s `RoadNetwork.segments` /
 * `get_nearby_segments`. No KDTree: candidate lists here are one bounding box's
 * worth of `drive`-network OSM ways, a few thousand at most, and this is called
 * once per blackout (corridor construction) plus once per blackout cycle
 * (progress matching, itself windowed to +/-20 m of arc length) - a linear scan is
 * not the cost centre a phone CPU needs to be protected from here. */
class RoadNetwork(val segments: List<RoadSegment>) {

    data class Candidate(val segment: RoadSegment, val index: Int, val proj: DoubleArray, val dist: Double)

    /** Segments within [radius] metres of [pos], nearest first. Mirrors
     * `map_provider.py`'s `get_nearby_segments`: project [pos] onto each segment's
     * line, clamp to the segment, keep it if the projected distance is inside
     * radius. */
    fun getNearbySegments(pos: DoubleArray, radius: Double): List<Candidate> {
        val out = ArrayList<Candidate>()
        for ((index, seg) in segments.withIndex()) {
            val l2 = seg.length * seg.length
            if (l2 == 0.0) continue
            val dx = seg.p2[0] - seg.p1[0]
            val dy = seg.p2[1] - seg.p1[1]
            var t = ((pos[0] - seg.p1[0]) * dx + (pos[1] - seg.p1[1]) * dy) / l2
            t = min(1.0, max(0.0, t))
            val proj = doubleArrayOf(seg.p1[0] + t * dx, seg.p1[1] + t * dy)
            val dist = hypot(pos[0] - proj[0], pos[1] - proj[1])
            if (dist < radius) out.add(Candidate(seg, index, proj, dist))
        }
        out.sortBy { it.dist }
        return out
    }
}

/**
 * A continuous road polyline with along-track (arc-length) parameterisation,
 * chained from a starting segment by greedily attaching whichever remaining
 * segment's start point lands within 5 m of the current chain end. Direct port of
 * `corridor_filter.py`'s `CorridorRoad.__init__` / `interpolate_position` /
 * `get_curvature_profile`.
 *
 * This is deliberately not real map matching: at a junction it grabs whichever
 * connected segment happens to be nearest within 5 m, which may be a crossing or
 * turning road rather than "straight ahead" - the Python reference has the same
 * limitation (`best_dist = 5.0` with no heading-continuity preference), reproduced
 * here rather than "fixed" out from under the validated scorecard numbers it was
 * measured against.
 */
class CorridorRoad(allSegments: List<RoadSegment>, initialIndex: Int) {

    /** [North, East] vertices of the chained polyline, in order. */
    val points: List<DoubleArray>
    private val segLengths: DoubleArray
    private val cumLengths: DoubleArray
    private val segHeadings: DoubleArray
    val totalLength: Double

    init {
        val initial = allSegments[initialIndex]
        val pts = ArrayList<DoubleArray>()
        pts.add(initial.p1)
        pts.add(initial.p2)
        var currP = initial.p2

        val remaining = ArrayList<RoadSegment>()
        for ((i, s) in allSegments.withIndex()) if (i != initialIndex) remaining.add(s)

        while (remaining.isNotEmpty()) {
            var bestI = -1
            var bestDist = 5.0
            for (i in remaining.indices) {
                val d1 = hypot(remaining[i].p1[0] - currP[0], remaining[i].p1[1] - currP[1])
                if (d1 < bestDist) {
                    bestDist = d1
                    bestI = i
                }
            }
            if (bestI < 0) break
            val next = remaining.removeAt(bestI)
            pts.add(next.p2)
            currP = next.p2
        }

        points = pts
        val n = pts.size - 1
        segLengths = DoubleArray(n)
        segHeadings = DoubleArray(n)
        cumLengths = DoubleArray(n + 1)
        for (i in 0 until n) {
            val dx = pts[i + 1][0] - pts[i][0]
            val dy = pts[i + 1][1] - pts[i][1]
            segLengths[i] = hypot(dx, dy)
            segHeadings[i] = atan2(dy, dx)
            cumLengths[i + 1] = cumLengths[i] + segLengths[i]
        }
        totalLength = cumLengths[n]
    }

    /** Arc length `s` (metres) -> ([North, East], heading). Clamped to
     * [0, totalLength], matching `interpolate_position`'s `np.clip`. */
    fun interpolatePosition(s: Double): Pair<DoubleArray, Double> {
        val sClamped = min(totalLength, max(0.0, s))
        var idx = cumLengths.indexOfLast { it <= sClamped }
        idx = min(max(0, idx), segLengths.size - 1)

        val segS = sClamped - cumLengths[idx]
        var t = segS / (segLengths[idx] + 1e-8)
        t = min(1.0, max(0.0, t))

        val p0 = points[idx]
        val p1 = points[idx + 1]
        val pos = doubleArrayOf(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
        return Pair(pos, segHeadings[idx])
    }

    /** Unwrapped heading at 1 m arc-length steps, for [RoadProgressMatcher]'s
     * cumulative-turn signature. Matches `get_curvature_profile`'s `s_grid` /
     * `headings_unwrapped` (the `curvatures` array itself is computed by the
     * Python reference but never consumed by anything downstream, so it is not
     * reproduced here). */
    fun headingProfile(ds: Double = 1.0): Pair<DoubleArray, DoubleArray> {
        val count = max(1, (totalLength / ds).toInt())
        val sGrid = DoubleArray(count) { it * ds }
        val headings = DoubleArray(count) { interpolatePosition(sGrid[it]).second }
        // np.unwrap: keep each step within (-pi, pi] of the previous one.
        val unwrapped = DoubleArray(count)
        if (count > 0) unwrapped[0] = headings[0]
        for (i in 1 until count) {
            var delta = headings[i] - headings[i - 1]
            while (delta > Math.PI) delta -= 2.0 * Math.PI
            while (delta < -Math.PI) delta += 2.0 * Math.PI
            unwrapped[i] = unwrapped[i - 1] + delta
        }
        return Pair(sGrid, unwrapped)
    }
}

/**
 * 1-D Kalman filter along a [CorridorRoad]. State `[s, v, b_v]`: along-track
 * distance (m), forward speed (m/s), forward-speed bias (m/s). Direct port of
 * `corridor_filter.py`'s `CorridorFilter` - same state, same F/Q, same H rows,
 * same clamp-speed-nonnegative-after-every-write behaviour.
 */
class CorridorFilterState(private val road: CorridorRoad, sInit: Double, vInit: Double) {
    // [s, v, b_v]
    var x: DoubleArray = doubleArrayOf(sInit, vInit, 0.0)
        private set
    var p: Array<DoubleArray> = arrayOf(
        doubleArrayOf(4.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(0.0, 0.0, 0.1)
    )
        private set

    private val qS = 0.01 * 0.01
    private val qV = 1.0 * 1.0
    private val qB = 0.01 * 0.01

    fun predict(dt: Double) {
        // F = [[1, dt, dt], [0, 1, 0], [0, 0, 1]]
        val newX = doubleArrayOf(
            x[0] + dt * x[1] + dt * x[2],
            x[1],
            x[2]
        )
        newX[1] = max(0.0, newX[1])

        // P' = F P F^T + Q, F as above, Q = diag(qS*dt, qV*dt, qB*dt)
        val f = arrayOf(
            doubleArrayOf(1.0, dt, dt),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0)
        )
        val fp = LinAlg.matMul(f, p)
        val fpFt = LinAlg.matMul(fp, LinAlg.transpose(f))
        fpFt[0][0] += qS * dt
        fpFt[1][1] += qV * dt
        fpFt[2][2] += qB * dt

        x = newX
        p = fpFt
    }

    /** `z_v = v + b_v`. [uncertaintyMeas] is a variance (not a 1-sigma), matching
     * `update_speed`'s `R = uncertainty_meas + safety_floor` - see
     * [Stage12Output.rChannelA]'s doc for why that quantity is already a variance
     * on the Stage 12 side. */
    fun updateSpeed(speedMeas: Double, uncertaintyMeas: Double, safetyFloor: Double = 0.25) {
        // H = [0, 1, 1]
        val hx = x[1] + x[2]
        val r = uncertaintyMeas + safetyFloor
        val s = p[1][1] + p[1][2] + p[2][1] + p[2][2] + r
        val y = speedMeas - hx

        val k = doubleArrayOf((p[0][1] + p[0][2]) / s, (p[1][1] + p[1][2]) / s, (p[2][1] + p[2][2]) / s)
        x = doubleArrayOf(x[0] + k[0] * y, x[1] + k[1] * y, x[2] + k[2] * y)
        x[1] = max(0.0, x[1])

        // P = (I - K H) P, H = [0, 1, 1] - K H's column c is k[row] for c in {1,2}, 0 for c=0.
        val ikh = Array(3) { row -> DoubleArray(3) { col -> (if (row == col) 1.0 else 0.0) - k[row] * (if (col == 1 || col == 2) 1.0 else 0.0) } }
        p = LinAlg.matMul(ikh, p)
    }

    /** `z_s = s`. Gated on `confidence >= 0.5` inside this call, matching
     * `update_progress_anchor`'s own internal gate (in addition to whatever
     * threshold the caller applies before calling this at all - see
     * [Stage12Channel]-style layered gating rationale in [CorridorChannel]). */
    fun updateProgressAnchor(sAnchor: Double, confidence: Double, baseStd: Double = 3.0) {
        if (confidence < 0.5) return
        val rS = (baseStd / max(1e-3, confidence)).let { it * it }
        // H = [1, 0, 0]
        val y = sAnchor - x[0]
        val s = p[0][0] + rS
        val k = doubleArrayOf(p[0][0] / s, p[1][0] / s, p[2][0] / s)
        x = doubleArrayOf(x[0] + k[0] * y, x[1] + k[1] * y, x[2] + k[2] * y)

        val ikh = Array(3) { row -> DoubleArray(3) { col -> (if (row == col) 1.0 else 0.0) - k[row] * (if (col == 0) 1.0 else 0.0) } }
        p = LinAlg.matMul(ikh, p)
    }

    /** Metric [North, East] position and heading at the current along-track
     * estimate. */
    fun position(): Pair<DoubleArray, Double> = road.interpolatePosition(x[0])
}

/**
 * Deterministic road-geometry progress anchors (`corridor_filter.py`'s
 * `RoadProgressMatcher`, "Phase 13"): aligns the cumulative IMU heading change
 * since blackout start against the candidate road's own curvature/turn signature,
 * to find where along the road that turn pattern actually happened.
 */
class RoadProgressMatcher(road: CorridorRoad) {
    private val sGrid: DoubleArray
    private val cumTurnMap: DoubleArray

    init {
        val (grid, headings) = road.headingProfile(1.0)
        sGrid = grid
        val h0 = if (headings.isNotEmpty()) headings[0] else 0.0
        cumTurnMap = DoubleArray(headings.size) { headings[it] - h0 }
    }

    /** Given the current along-track estimate [sEst] and the cumulative IMU
     * heading change (rad) since blackout start, finds the nearest arc length
     * whose turn signature matches best within +/- [searchWindow] metres, and a
     * confidence that decays with the angular mismatch. Returns `(sEst, 0.0)` if
     * the search window falls entirely outside the road (matches the Python
     * reference's `if not np.any(mask): return s_est, 0.0`). */
    fun matchProgress(sEst: Double, imuHeadingChange: Double, searchWindow: Double = 20.0): Pair<Double, Double> {
        if (sGrid.isEmpty()) return Pair(sEst, 0.0)
        val sMin = max(0.0, sEst - searchWindow)
        val sMax = min(sGrid.last(), sEst + searchWindow)

        var bestIdx = -1
        var bestDiff = Double.MAX_VALUE
        for (i in sGrid.indices) {
            if (sGrid[i] < sMin || sGrid[i] > sMax) continue
            val diff = abs(cumTurnMap[i] - imuHeadingChange)
            if (diff < bestDiff) {
                bestDiff = diff
                bestIdx = i
            }
        }
        if (bestIdx < 0) return Pair(sEst, 0.0)

        val conf = exp(-0.5 * (bestDiff / 0.15) * (bestDiff / 0.15))
        return Pair(sGrid[bestIdx], conf)
    }
}

/** Where the one-shot road-network fetch is in its lifecycle. Display-only: nothing in
 * the fusion path branches on it (a missing network already just means "no road
 * locked" via [RoadNetworkProvider.currentNetwork] returning null), it exists so the
 * UI can say *why* the corridor is silent instead of leaving "no road locked" to
 * cover offline, still-loading and nothing-mapped-here alike. */
enum class RoadNetworkState {
    /** No fetch requested yet (no GNSS fix so far this session). */
    IDLE,
    /** First attempt in flight. */
    FETCHING,
    /** An earlier attempt failed and a later one is scheduled or in flight. */
    RETRYING,
    /** A network is available to [CorridorChannel.onBlackoutStart]. */
    READY,
    /** Every attempt failed; the corridor stays silent for the rest of the session. */
    FAILED
}

data class RoadNetworkStatus(
    val state: RoadNetworkState,
    /** Segments in the fetched network; 0 unless [state] is [RoadNetworkState.READY]. */
    val segmentCount: Int = 0,
    /** Attempts so far: 0 while IDLE, 1 while FETCHING, the number that have already
     * FAILED while RETRYING, and the number used in total for READY / FAILED. */
    val attempt: Int = 0
)

/** The OSM-fetching boundary. Implemented against Overpass in the sensors package
 * (`OverpassRoadNetworkProvider`); kept as an interface here so [CorridorChannel]
 * is testable without network access, same split as [Stage12SpeedModel]. */
interface RoadNetworkProvider {
    /** Kick off an async fetch of the drivable road network within [radiusM] of
     * ([lat], [lon]), projected into [frame]. Non-blocking - call [currentNetwork]
     * to see whether a result is ready yet. A sane implementation ignores repeat
     * calls close to its last fetch centre rather than re-querying Overpass on
     * every GNSS fix; [CorridorChannel] calls this once, on the session's first
     * fix, and relies on that. */
    fun requestAround(lat: Double, lon: Double, frame: LocalFrame, radiusM: Double = 3000.0)

    /** The most recently completed fetch's network, or null if none has completed
     * yet (still pending, or every attempt so far failed). Never throws. */
    fun currentNetwork(): RoadNetwork?

    /** Display-only fetch progress. The default derives READY/IDLE from
     * [currentNetwork] alone, so a provider that cannot report retries or failures
     * (a test fake, say) needs no changes. */
    fun status(): RoadNetworkStatus {
        val network = currentNetwork()
        return if (network != null) {
            RoadNetworkStatus(RoadNetworkState.READY, network.segments.size, attempt = 1)
        } else {
            RoadNetworkStatus(RoadNetworkState.IDLE)
        }
    }
}

/** Output of one blackout's corridor tracking, ready to hand to [DualChannelUkf.step]
 * as a road-signature position correction. */
data class CorridorOutput(
    val positionNorthEast: DoubleArray,
    val confidence: Double,
    /** Distance between the corridor's road-snapped position and the outer UKF's
     * own free-space position estimate this cycle - a display number only (how
     * hard the road match is pulling), not fed back into anything. Null on the
     * first cycle after a road locks on, before the caller has an outer estimate
     * to compare against. */
    val correctionMagnitudeM: Double?
)

/**
 * Orchestrates road-network selection, the 1-D corridor filter, and progress-anchor
 * matching for one blackout. One instance per [FusionPipeline] run, mirroring
 * [Stage12Channel]'s lifecycle: constructed once (here, immediately - it has no
 * leveling dependency), fed every RUNNING cycle, but only produces output during a
 * blackout with a road actually locked on.
 *
 * Bridges `canonical_evaluate_v4.py`'s "Mode F: Best Adaptive Hybrid" path:
 * `predict` -> `update_speed` -> track cumulative IMU turn -> `match_progress` ->
 * `update_progress_anchor` when confidence clears the internal 0.65 gate (matching
 * the Python reference's HYBRID threshold exactly - a separate, stricter threshold
 * is applied by [FusionPipeline] before this channel's output reaches the outer
 * UKF; see that call site for why two thresholds is the deliberate choice here).
 *
 * The road network itself is fetched once, opportunistically, as soon as the
 * session has its first GNSS fix (see [RoadNetworkProvider]) - not scoped to the
 * eventual blackout location, which is not known in advance on a live phone the
 * way `canonical_evaluate_v4.py`'s offline chain (which sees the whole recorded
 * route before picking a bounding box) can assume. A blackout starting before that
 * fetch completes, or in an area Overpass has no `drive`-network data for, simply
 * finds no candidate road and this channel produces nothing - [FusionPipeline]
 * keeps running on Channel P / Stage 12 and the outer UKF's own process model
 * exactly as it would if this channel did not exist.
 */
class CorridorChannel(private val roadNetworkProvider: RoadNetworkProvider) {

    private var requestedFetch = false
    private var corridorFilter: CorridorFilterState? = null
    private var roadMatcher: RoadProgressMatcher? = null
    private var lockedRoad: CorridorRoad? = null
    private var cumImuTurn = 0.0

    /** Vertices ([North, East] metres) of the chained road polyline this blackout is
     * tracking along, or null when no road is locked. Same list instance for the whole
     * blackout, so a caller can hold it by reference without copying per cycle. Callers
     * must treat it as read-only. */
    val lockedRoadPoints: List<DoubleArray>? get() = lockedRoad?.points

    /** Progress of the road-network fetch behind this channel, display-only. */
    fun networkStatus(): RoadNetworkStatus = roadNetworkProvider.status()

    /** Whether a candidate road is currently locked on for this blackout (i.e.
     * [update] can produce output). False before the first blackout, after a
     * blackout ends (see [onBlackoutEnd]), or if no candidate road was found. */
    val roadLocked: Boolean get() = corridorFilter != null

    /** Call once, as soon as [FusionPipeline] has its first GNSS fix and local
     * frame. Fires the road-network fetch at most once per session - repeat calls
     * are the caller's problem to avoid, not this class's (matches
     * [RoadNetworkProvider]'s own "ignore close repeats" contract, belt and
     * braces). */
    fun onFirstFix(lat: Double, lon: Double, frame: LocalFrame) {
        if (requestedFetch) return
        requestedFetch = true
        roadNetworkProvider.requestAround(lat, lon, frame)
    }

    /** Call once, at the instant a blackout starts. Picks the nearest candidate
     * segment to [pos] (plain nearest, matching `canonical_evaluate_v4.py`'s
     * actually-executed selection - `build_corridor_from_map`'s heading-scored
     * variant is defined in the Python reference but never called by its own
     * evaluator, so it is not reproduced here either), chains it into a
     * [CorridorRoad], and seeds the corridor filter at s=0 with [initialSpeed].
     * Leaves [roadLocked] false (this channel silent for the whole blackout) if no
     * road network has finished fetching yet, or nothing is within 50 m of [pos] -
     * matching `get_nearby_segments(pos_start, radius=50.0)`. */
    fun onBlackoutStart(pos: DoubleArray, initialSpeed: Double) {
        cumImuTurn = 0.0
        corridorFilter = null
        roadMatcher = null
        lockedRoad = null

        val network = roadNetworkProvider.currentNetwork() ?: return
        val candidates = network.getNearbySegments(pos, radius = 50.0)
        val best = candidates.minByOrNull { it.dist } ?: return

        val road = CorridorRoad(network.segments, initialIndex = best.index)
        lockedRoad = road
        roadMatcher = RoadProgressMatcher(road)
        corridorFilter = CorridorFilterState(road, sInit = 0.0, vInit = max(0.0, initialSpeed))
    }

    /** No road for the next blackout until [onBlackoutStart] locks one on again -
     * a road-lock from one blackout has no reason to carry into a different one,
     * especially since the vehicle kept moving (off GNSS) through the whole
     * reacquisition window in between. */
    fun onBlackoutEnd() {
        corridorFilter = null
        roadMatcher = null
        lockedRoad = null
    }

    /** Call once per UKF cycle during a blackout, after the outer filter's own
     * Channel A speed and yaw rate for this cycle are known. [speedMeas] /
     * [speedUncertainty] are whatever [FusionPipeline] is already feeding Channel
     * A this cycle (Stage 12 if trusted, else Channel P) - the corridor filter's
     * own `update_speed` measurement is the same evidence, not a separate source.
     * [yawRateRadS] is the same corrected yaw rate the outer filter's process
     * model uses this cycle. [outerPositionNorthEast] is the outer UKF's own
     * current position estimate, used only to compute the display-only
     * `correctionMagnitudeM` on the returned output. Returns null if no road is
     * locked on (nothing to update). */
    fun update(
        dt: Double,
        speedMeas: Double,
        speedUncertainty: Double,
        yawRateRadS: Double,
        outerPositionNorthEast: DoubleArray
    ): CorridorOutput? {
        val filter = corridorFilter ?: return null
        val matcher = roadMatcher ?: return null

        filter.predict(dt)
        filter.updateSpeed(speedMeas, speedUncertainty)
        cumImuTurn += yawRateRadS * dt

        val (sAnchor, conf) = matcher.matchProgress(filter.x[0], cumImuTurn)
        // 0.65: matches canonical_evaluate_v4.py's HYBRID mode gate exactly (the
        // scorecard's best-performing configuration). This is the corridor
        // filter's OWN internal gate on trusting a progress anchor; FusionPipeline
        // applies a separate, stricter gate before this channel's output is
        // allowed to correct the outer UKF's position - see that call site.
        if (conf > 0.65) filter.updateProgressAnchor(sAnchor, conf)

        val (pos, _) = filter.position()
        val correction = hypot(pos[0] - outerPositionNorthEast[0], pos[1] - outerPositionNorthEast[1])
        return CorridorOutput(pos, conf, correction)
    }
}
