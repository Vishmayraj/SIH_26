package org.sih26.deadreckoning.sensors

import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import kotlin.math.cos
import org.json.JSONArray
import org.json.JSONObject
import org.sih26.deadreckoning.fusion.LocalFrame
import org.sih26.deadreckoning.fusion.RoadNetwork
import org.sih26.deadreckoning.fusion.RoadNetworkProvider
import org.sih26.deadreckoning.fusion.RoadNetworkState
import org.sih26.deadreckoning.fusion.RoadNetworkStatus
import org.sih26.deadreckoning.fusion.RoadSegment

private const val TAG = "OverpassRoadNetwork"
private const val OVERPASS_URL = "https://overpass-api.de/api/interpreter"

/** Wait before the 2nd, 3rd and 4th attempt. A session that starts before mobile data
 * has attached (or in a dead spot) should still get a road network once connectivity
 * returns, but hammering the public Overpass instance is not acceptable behaviour
 * either, hence a short bounded ladder rather than a fixed-interval loop. */
private val RETRY_DELAYS_S = longArrayOf(10, 30, 60)

/**
 * Overpass API-backed [RoadNetworkProvider]: the only Android-specific piece of the
 * corridor bridge (see fusion/CorridorFilter.kt's class docs for everything else),
 * same split as MotionSpeedNetOnnx / Stage12Channel.
 *
 * Stands in for `tools/map_matching/map_provider.py`'s `RoadNetwork`, which uses
 * `osmnx` to download the same `drive`-network graph and cache it as `.graphml` on
 * disk. There is no osmnx (or any OSM graph library) for Android, and no offline
 * extract shipped with this app - `map_matching/README.md` says as much: a real OSM
 * extract for at least one corridor was always a prerequisite this scaffolding was
 * waiting on, not something baked into the repo. This queries Overpass directly
 * instead, general-purpose rather than tied to one region.
 *
 * Highway-type filter mirrors osmnx's `network_type='drive'` preset (public roads a
 * car can drive on - excludes footways, cycleways, unclassified-but-pedestrian
 * paths, etc; includes the `_link` slip-road variants).
 *
 * Fetches once per session (`requestAround` no-ops on any call after the first), on a
 * single background thread, retrying a failed attempt up to three more times on a
 * 10 s / 30 s / 60 s ladder, and never throws back into the caller - every failure
 * mode (no connectivity, Overpass timeout/rate limit, malformed response) leaves
 * `currentNetwork()` returning null, with [status] saying whether that is "still
 * trying" or "gave up". `CorridorChannel` treats a null network exactly like "not
 * fetched yet": the channel stays silent and `FusionPipeline` runs on Channel P /
 * Stage 12 alone, same as if this provider had never been constructed.
 */
class OverpassRoadNetworkProvider : RoadNetworkProvider {

    private val executor = Executors.newSingleThreadScheduledExecutor()

    @Volatile private var fetchStarted = false

    @Volatile private var network: RoadNetwork? = null

    @Volatile private var status = RoadNetworkStatus(RoadNetworkState.IDLE)

    override fun requestAround(lat: Double, lon: Double, frame: LocalFrame, radiusM: Double) {
        // Single-flight: the first caller (FusionPipeline.onGnss, on the session's
        // first fix - see CorridorChannel.onFirstFix) wins, everything after that
        // is a no-op. A live re-centre on a long drive that wanders outside the
        // original radius is a real gap (see the class-level "what's deliberately
        // not done" note in the handoff), not something this guard is trying to
        // paper over.
        if (fetchStarted) return
        fetchStarted = true
        status = RoadNetworkStatus(RoadNetworkState.FETCHING, attempt = 1)
        try {
            executor.execute { runAttempt(lat, lon, frame, radiusM, attempt = 1) }
        } catch (e: RejectedExecutionException) {
            // close() already ran (session stopped between the first fix and here).
            status = RoadNetworkStatus(RoadNetworkState.FAILED, attempt = 1)
        }
    }

    /** One fetch attempt, always on the executor thread. A failure schedules the next
     * attempt on the same thread after [RETRY_DELAYS_S], or gives up for the session
     * once the ladder is exhausted. Never throws. */
    private fun runAttempt(lat: Double, lon: Double, frame: LocalFrame, radiusM: Double, attempt: Int) {
        status = if (attempt == 1) {
            RoadNetworkStatus(RoadNetworkState.FETCHING, attempt = 1)
        } else {
            RoadNetworkStatus(RoadNetworkState.RETRYING, attempt = attempt - 1)
        }
        try {
            val bbox = boundingBox(lat, lon, radiusM)
            val query = buildQuery(bbox)
            val json = postOverpassQuery(query)
            val segments = parseSegments(json, frame)
            network = RoadNetwork(segments)
            status = RoadNetworkStatus(RoadNetworkState.READY, segments.size, attempt)
            Log.i(TAG, "Corridor road network ready: ${segments.size} segments within ${radiusM.toInt()} m of start (attempt $attempt).")
        } catch (e: Exception) {
            val nextDelayS = RETRY_DELAYS_S.getOrNull(attempt - 1)
            if (nextDelayS == null) {
                status = RoadNetworkStatus(RoadNetworkState.FAILED, attempt = attempt)
                Log.e(TAG, "Road network fetch failed after $attempt attempts; corridor matching disabled for this session.", e)
                return
            }
            status = RoadNetworkStatus(RoadNetworkState.RETRYING, attempt = attempt)
            Log.w(TAG, "Road network fetch attempt $attempt failed; retrying in ${nextDelayS}s.", e)
            try {
                // Explicit Runnable: a bare lambda is ambiguous between the Runnable and
                // Callable overloads of schedule().
                executor.schedule(
                    Runnable { runAttempt(lat, lon, frame, radiusM, attempt + 1) },
                    nextDelayS, TimeUnit.SECONDS
                )
            } catch (rejected: RejectedExecutionException) {
                // close() ran while this attempt was in flight - the session is over.
                status = RoadNetworkStatus(RoadNetworkState.FAILED, attempt = attempt)
            }
        }
    }

    /** Cancels any pending retry and releases the fetch thread. Idempotent. Called by
     * the recording service when a session ends: this provider is constructed per
     * session, so without it every recording would leave one idle thread behind (and
     * a pending retry could fire after the session it was for had already ended). */
    fun close() {
        executor.shutdownNow()
    }

    override fun currentNetwork(): RoadNetwork? = network

    override fun status(): RoadNetworkStatus = status

    /** [south, west, north, east] in degrees, padded generously around ([lat],
     * [lon]) - matches map_provider.py's own 0.01-degree pad philosophy, just
     * expressed as a metre radius instead since a live session does not know its
     * eventual bounding box the way an offline session (which sees the whole
     * recorded route first) does. */
    private fun boundingBox(lat: Double, lon: Double, radiusM: Double): DoubleArray {
        val latDelta = radiusM / 111_320.0
        val lonDelta = radiusM / (111_320.0 * cos(Math.toRadians(lat)).coerceAtLeast(0.01))
        return doubleArrayOf(lat - latDelta, lon - lonDelta, lat + latDelta, lon + lonDelta)
    }

    /** Overpass QL requesting geometry inline (`out geom;`) so no separate node
     * lookup is needed - each way's `geometry` array is already lat/lon pairs in
     * order. `highway` filter mirrors osmnx's `network_type='drive'`. */
    private fun buildQuery(bbox: DoubleArray): String {
        val (south, west, north, east) = bbox
        val highwayTypes = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|" +
            "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|service"
        return """
            [out:json][timeout:25];
            (
              way["highway"~"^($highwayTypes)$"]["area"!="yes"]($south,$west,$north,$east);
            );
            out geom;
        """.trimIndent()
    }

    private fun postOverpassQuery(query: String): JSONObject {
        val url = URL(OVERPASS_URL)
        val connection = url.openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")

            val body = "data=" + URLEncoder.encode(query, "UTF-8")
            connection.outputStream.use { out: OutputStream -> out.write(body.toByteArray(Charsets.UTF_8)) }

            val status = connection.responseCode
            if (status !in 200..299) {
                throw java.io.IOException("Overpass returned HTTP $status")
            }
            val text = BufferedReader(InputStreamReader(connection.inputStream, Charsets.UTF_8)).use { it.readText() }
            return JSONObject(text)
        } finally {
            connection.disconnect()
        }
    }

    /** Every consecutive pair of points in each way's `geometry` array becomes one
     * [RoadSegment], projected into [frame] - matches map_provider.py's
     * `_build_segments`' per-coordinate-pair loop, minus the edge_id bookkeeping
     * [RoadSegment]'s doc explains is unused by anything this app ports. */
    private fun parseSegments(response: JSONObject, frame: LocalFrame): List<RoadSegment> {
        val elements = response.optJSONArray("elements") ?: JSONArray()
        val segments = ArrayList<RoadSegment>()
        for (i in 0 until elements.length()) {
            val element = elements.getJSONObject(i)
            if (element.optString("type") != "way") continue
            val geometry = element.optJSONArray("geometry") ?: continue

            var prev: DoubleArray? = null
            for (j in 0 until geometry.length()) {
                val point = geometry.getJSONObject(j)
                val lat = point.optDouble("lat", Double.NaN)
                val lon = point.optDouble("lon", Double.NaN)
                if (lat.isNaN() || lon.isNaN()) continue
                val projected = frame.toNorthEast(lat, lon)

                val previous = prev
                if (previous != null) {
                    segments.add(RoadSegment(previous, projected))
                }
                prev = projected
            }
        }
        return segments
    }
}
