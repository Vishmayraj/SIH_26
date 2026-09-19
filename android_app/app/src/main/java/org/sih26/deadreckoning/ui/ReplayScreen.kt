package org.sih26.deadreckoning.ui

import android.net.Uri
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import java.io.FileInputStream
import java.util.Locale
import kotlin.coroutines.cancellation.CancellationException
import kotlin.math.hypot
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.sih26.deadreckoning.R
import org.sih26.deadreckoning.replay.ReplayLoader
import org.sih26.deadreckoning.replay.ReplayPoint
import org.sih26.deadreckoning.replay.ReplaySession
import org.sih26.deadreckoning.replay.ReplaySource
import org.sih26.deadreckoning.replay.ReplayTrack
import org.sih26.deadreckoning.sessions.SessionRecord

/**
 * Plays a recorded drive back on the same [TrajectoryCanvas] the Live screen uses.
 *
 * Two sources, one screen: a phone recording (from the Sessions list, or any `.jsonl`
 * picked from storage) and an IO-VNBD smartphone CSV picked from storage. What plays
 * is what the file contains - GNSS truth for both, plus the recording app's own fused
 * and coast tracks for a phone session. Nothing is re-run through the filter here.
 *
 * [session] is owned by the caller so a loaded drive survives switching to another
 * screen from the navigation drawer and back.
 */
@Composable
fun ReplayScreen(
    records: List<SessionRecord>,
    session: ReplaySession?,
    onSession: (ReplaySession?) -> Unit,
    pickFile: () -> Unit,
    pickedUri: Uri?,
    consumePickedUri: () -> Unit,
    loadFromUri: (Uri) -> ReplaySession
) {
    var loading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    // Parsing a multi-megabyte log is real work: off the main thread, and the screen
    // says so instead of looking frozen.
    fun load(block: () -> ReplaySession) {
        scope.launch {
            loading = true
            error = null
            try {
                onSession(withContext(Dispatchers.IO) { block() })
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                error = e.message ?: e.javaClass.simpleName
            } finally {
                loading = false
            }
        }
    }

    LaunchedEffect(pickedUri) {
        val uri = pickedUri ?: return@LaunchedEffect
        consumePickedUri()
        load { loadFromUri(uri) }
    }

    if (session != null) {
        ReplayPlayer(session, onClose = { onSession(null) })
        return
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            AppMark(56.dp)
            Column {
                Text("Replay a drive", style = MaterialTheme.typography.titleLarge)
                Text(
                    "Play back a recording on the trajectory view.",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }

        Button(onClick = pickFile, enabled = !loading, modifier = Modifier.fillMaxWidth()) {
            Text("Open a file (phone .jsonl or IO-VNBD .csv)")
        }
        Text(
            "IO-VNBD replay needs the smartphone (S-) file: it plays the GPS track from the " +
                "GPS latitude/longitude columns.",
            style = MaterialTheme.typography.bodySmall
        )

        if (loading) LinearProgressIndicator(Modifier.fillMaxWidth())
        error?.let {
            Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                Text(it, Modifier.padding(12.dp), color = MaterialTheme.colorScheme.onErrorContainer)
            }
        }

        Text("Recorded on this phone", style = MaterialTheme.typography.titleSmall)
        if (records.isEmpty()) {
            Text("No completed sessions yet. Recordings appear here after you stop them.", style = MaterialTheme.typography.bodyMedium)
        }
        records.forEach { r ->
            Card(Modifier.fillMaxWidth().clickable(enabled = !loading) {
                load { FileInputStream(r.rawFile).use { ReplayLoader.load(it, "Drive ${r.startedUtc.take(16).replace('T', ' ')} UTC") } }
            }) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("Drive - ${r.startedUtc.take(16).replace('T', ' ')} UTC", style = MaterialTheme.typography.titleSmall)
                    Text("${duration(r.durationS)} - ${fmt(r.distanceM / 1000)} km - tap to replay", style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

/** The launcher icon's own two vector layers, clipped to a circle: reuses the app's
 * existing drawables rather than adding a second piece of artwork. */
@Composable
private fun AppMark(size: Dp) {
    Box(Modifier.size(size).clip(CircleShape)) {
        Image(painterResource(R.drawable.ic_launcher_background), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        Image(painterResource(R.drawable.ic_launcher_foreground), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
    }
}

private val SPEEDS = listOf(1f, 5f, 10f, 30f)

@Composable
private fun ReplayPlayer(session: ReplaySession, onClose: () -> Unit) {
    var t by remember(session) { mutableStateOf(session.startS) }
    var playing by remember(session) { mutableStateOf(true) }
    var speed by remember(session) { mutableStateOf(5f) }

    // Advances the playhead in wall-clock time, so 5x is 5x whatever the frame rate.
    LaunchedEffect(session, playing, speed) {
        if (!playing) return@LaunchedEffect
        if (t >= session.endS) t = session.startS
        var last = System.nanoTime()
        while (true) {
            delay(50)
            val now = System.nanoTime()
            val next = t + (now - last) / 1e9 * speed
            last = now
            if (next >= session.endS) {
                t = session.endS
                playing = false
                return@LaunchedEffect
            }
            t = next
        }
    }

    // Built once per session. "Everything up to t" is then a prefix view of this list,
    // not a per-frame rebuild.
    val allPoints = remember(session) { session.timeline.map { it.toTrackPoint() } }
    val visible = allPoints.subList(0, session.timelineCountUpTo(t))

    val truthNow = session.truthAt(t)
    val fusedNow = session.fusedAt(t)
    val coastNow = session.coastAt(t)
    // The last fix is at most ~1 s old, so this is a live indication, not a metric: the
    // number that goes on a slide is computed offline from the raw log.
    fun gapToTruth(p: ReplayPoint?): Double? =
        if (p != null && truthNow != null && t - p.t < 1.0) hypot(p.north - truthNow.north, p.east - truthNow.east) else null

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                session.label, Modifier.weight(1f),
                style = MaterialTheme.typography.titleSmall, maxLines = 2
            )
            OutlinedButton(onClick = onClose) { Text("Close") }
        }

        if (truthNow?.withheld == true) {
            Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                Text(
                    "GNSS BLACKOUT - fixes were withheld from the filter here",
                    Modifier.padding(12.dp), style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }
        }

        TrajectoryCanvas(
            points = visible,
            modifier = Modifier.fillMaxWidth(),
            extentPoints = allPoints,
            emptyMessage = "Press play - the track builds up as the drive is replayed."
        )

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (session.endS > session.startS) {
                    Slider(
                        value = t.toFloat(),
                        onValueChange = { t = it.toDouble() },
                        valueRange = session.startS.toFloat()..session.endS.toFloat()
                    )
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    IconButton(onClick = { playing = !playing }) {
                        Icon(if (playing) Icons.Default.Pause else Icons.Default.PlayArrow, contentDescription = if (playing) "Pause" else "Play")
                    }
                    IconButton(onClick = { t = session.startS; playing = true }) {
                        Icon(Icons.Default.Replay, contentDescription = "Restart")
                    }
                    Text(
                        "${duration(t - session.startS)} / ${duration(session.endS - session.startS)}",
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    SPEEDS.forEach { s ->
                        val modifier = Modifier.weight(1f)
                        if (s == speed) {
                            FilledTonalButton({ speed = s }, modifier, contentPadding = PaddingValues(horizontal = 8.dp)) { Text("${s.toInt()}x") }
                        } else {
                            OutlinedButton({ speed = s }, modifier, contentPadding = PaddingValues(horizontal = 8.dp)) { Text("${s.toInt()}x") }
                        }
                    }
                }
            }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                ReadoutRow(
                    "GNSS speed", truthNow?.speedMps?.let { "${fmt(it * 3.6)} km/h" } ?: "--",
                    "Heading", truthNow?.headingDeg?.let { "${fmt(it)} deg" } ?: "--"
                )
                ReadoutRow(
                    "GNSS accuracy", truthNow?.accuracyM?.let { "${fmt(it)} m" } ?: "--",
                    if (session.source == ReplaySource.IOVNBD_CSV) "Satellites" else "GNSS",
                    if (session.source == ReplaySource.IOVNBD_CSV) (truthNow?.satellites?.toString() ?: "--")
                    else if (truthNow?.withheld == true) "withheld" else "live"
                )
                if (session.hasFusion) {
                    ReadoutRow(
                        "Fused vs last fix", gapToTruth(fusedNow)?.let { "${fmt(it)} m" } ?: "--",
                        "Coast vs last fix", gapToTruth(coastNow)?.let { "${fmt(it)} m" } ?: "--"
                    )
                    Text(
                        "Gaps are measured against the most recent GNSS fix, so they are only meaningful " +
                            "during a blackout, and are a live indication rather than the reported drift.",
                        style = MaterialTheme.typography.labelSmall
                    )
                } else {
                    Text(
                        "GPS track only: this file carries no fused or coast output to replay.",
                        style = MaterialTheme.typography.labelSmall
                    )
                }
            }
        }
    }
}

private fun ReplayPoint.toTrackPoint() = TrackPoint(
    north, east,
    when (track) {
        ReplayTrack.TRUTH -> TrackKind.TRUTH
        ReplayTrack.FUSED -> TrackKind.FUSED
        ReplayTrack.COAST -> TrackKind.COAST
    }
)

@Composable
private fun ReadoutRow(label1: String, value1: String, label2: String, value2: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Column(Modifier.weight(1f)) { Text(label1, style = MaterialTheme.typography.labelSmall); Text(value1, style = MaterialTheme.typography.bodyLarge) }
        Column(Modifier.weight(1f)) { Text(label2, style = MaterialTheme.typography.labelSmall); Text(value2, style = MaterialTheme.typography.bodyLarge) }
    }
}

private fun fmt(value: Double) = String.format(Locale.ROOT, "%.2f", value)
private fun duration(seconds: Double): String {
    val s = seconds.coerceAtLeast(0.0)
    return "%dm %02ds".format(Locale.ROOT, (s / 60).toInt(), (s % 60).toInt())
}
