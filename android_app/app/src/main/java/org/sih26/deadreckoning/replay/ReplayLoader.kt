package org.sih26.deadreckoning.replay

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import org.sih26.deadreckoning.fusion.LocalFrame

/**
 * Reads the two things the Replay page can play:
 *
 *  - a phone recording in this app's own JSONL format (`SessionLogger`), whether it
 *    came from the Sessions list or was exported from another phone, and
 *  - an IO-VNBD smartphone ("S-") CSV.
 *
 * The format is sniffed from the first line, so callers never have to say which.
 *
 * The JSONL parser only ever looks at `gnss`, `fused` and `coast` lines and skips the
 * 100 Hz `imu` lines by prefix without parsing them: a recording is dominated by IMU
 * lines, and nothing on this page uses them. Numbers are pulled out by key rather than
 * through a JSON library because the logger writes a fixed, flat schema and the
 * `fused`/`coast` streams are 100 Hz each - allocating a JSON object per line to read
 * two doubles would be most of the load time.
 *
 * Failures throw [IllegalArgumentException] with a message meant to be shown to the
 * person as-is.
 */
object ReplayLoader {

    /** Fused/coast are logged every IMU cycle (100 Hz); the trajectory view has no use
     * for more than this many points per second. */
    private const val MIN_TRACK_SPACING_S = 0.2

    fun load(input: InputStream, label: String): ReplaySession {
        val reader = BufferedReader(InputStreamReader(input, Charsets.UTF_8), 1 shl 16)
        reader.mark(1 shl 16)
        var first = reader.readLine()
        while (first != null && first.isBlank()) first = reader.readLine()
        require(first != null) { "The file is empty." }
        reader.reset()

        return if (first.trimStart('\uFEFF', ' ').startsWith("{")) {
            parseJsonl(reader, label)
        } else {
            parseIovnbdCsv(reader, label)
        }
    }

    // ---------------------------------------------------------------- phone JSONL

    private const val GNSS_PREFIX = "{\"type\":\"gnss\""
    private const val FUSED_PREFIX = "{\"type\":\"fused\""
    private const val COAST_PREFIX = "{\"type\":\"coast\""
    private const val T = "\"t\":"
    private const val PN = "\"pn\":"
    private const val PE = "\"pe\":"
    private const val LAT = "\"lat\":"
    private const val LON = "\"lon\":"
    private const val SPEED = "\"speed\":"
    private const val BEARING = "\"bearing\":"
    private const val ACCURACY = "\"accuracy\":"

    private fun parseJsonl(reader: BufferedReader, label: String): ReplaySession {
        var frame: LocalFrame? = null
        val truth = ArrayList<TruthFix>()
        val fused = ArrayList<ReplayPoint>()
        val coast = ArrayList<ReplayPoint>()
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
                // header, imu, anything unknown: not needed here.
            }
        }

        require(truth.isNotEmpty()) { "No GNSS fixes in this session, so there is no track to replay." }

        val starts = listOfNotNull(truth.first().t, fused.firstOrNull()?.t, coast.firstOrNull()?.t)
        val ends = listOfNotNull(truth.last().t, fused.lastOrNull()?.t, coast.lastOrNull()?.t)
        return ReplaySession(
            label = label, source = ReplaySource.PHONE_SESSION,
            startS = starts.min(), endS = ends.max(),
            truth = truth, fused = fused, coast = coast
        )
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

    private fun parseIovnbdCsv(reader: BufferedReader, label: String): ReplaySession {
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

        val truth = ArrayList<TruthFix>()
        var frame: LocalFrame? = null
        var firstT: Double? = null
        var lastRowT = 0.0
        var lastLat = Double.NaN
        var lastLon = Double.NaN

        while (true) {
            val line = reader.readLine() ?: break
            if (line.isBlank()) continue
            val f = splitCsv(line)
            fun cell(i: Int): Double? =
                if (i < 0) null else f.getOrNull(i)?.trim()?.toDoubleOrNull()?.takeIf { it.isFinite() }

            val tMs = cell(timeIdx) ?: continue
            val tAbs = tMs / 1000.0
            val t0 = firstT ?: tAbs.also { firstT = it }
            val t = tAbs - t0
            // Rows should be time-ordered; a row that runs backwards is dropped rather
            // than allowed to scramble the timeline.
            if (t < lastRowT) continue
            lastRowT = t

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

        return ReplaySession(
            label = label, source = ReplaySource.IOVNBD_CSV,
            startS = 0.0, endS = maxOf(lastRowT, truth.last().t),
            truth = truth, fused = emptyList(), coast = emptyList()
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
