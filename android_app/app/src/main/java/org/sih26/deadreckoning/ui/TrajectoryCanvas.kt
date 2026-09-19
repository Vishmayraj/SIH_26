package org.sih26.deadreckoning.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.unit.dp
import kotlin.math.max

/**
 * One track sample in the local tangent frame (meters, north/east), tagged by
 * source so the tracks the field-test milestone calls out - raw GNSS truth, the fused
 * (UKF) estimate, the no-correction coast baseline, and the corridor filter's
 * road-snapped position - stay visually and semantically separate. Nothing here decides what counts as "true"; it just
 * renders whatever [FusionSnapshot] already reports.
 */
enum class TrackKind { TRUTH, FUSED, COAST, CORRIDOR }

data class TrackPoint(val north: Double, val east: Double, val kind: TrackKind)

/**
 * A lightweight north-up trajectory view. Deliberately not a map: no tiles, no
 * network, no dependency risk this close to a deadline. It exists purely to make
 * "is the fused track actually tracking truth, and is that materially better than
 * doing nothing" visible at a glance during a blackout demo.
 *
 * [road] is the OSM road polyline the corridor filter locked onto (North/East metres,
 * from FusionSnapshot.corridorRoad). It is drawn as an underlay only and never widens
 * the view: the extent comes from the tracks alone, so a chained road that runs for
 * kilometres cannot shrink the track to a dot. Segments outside the view are skipped.
 *
 * [extentPoints], when given, sets the view from that list instead of from [points]:
 * a replay passes its whole session here so the view stays put while [points] grows,
 * instead of re-zooming on every frame. The legend likewise lists only the tracks
 * present in whichever list sets the view.
 */
@Composable
fun TrajectoryCanvas(
    points: List<TrackPoint>,
    modifier: Modifier = Modifier,
    road: List<DoubleArray>? = null,
    extentPoints: List<TrackPoint>? = null,
    emptyMessage: String = "Trajectory will appear once the fix is initialised."
) {
    val viewPoints = extentPoints ?: points
    val truthColor = Color(0xFF2E7D32)   // GNSS ground truth - green
    val fusedColor = Color(0xFF1565C0)   // UKF fused estimate - blue
    val coastColor = Color(0xFFE65100)   // no-correction coast baseline - orange
    val corridorColor = Color(0xFF8E24AA) // corridor road-snapped position - purple
    val roadColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f)
    val present = viewPoints.mapTo(HashSet()) { it.kind }
    val hasRoad = road != null && road.size >= 2

    Box(modifier.fillMaxWidth().height(220.dp).background(MaterialTheme.colorScheme.surfaceVariant)) {
        if (points.size < 2) {
            Text(
                emptyMessage,
                Modifier.align(Alignment.Center).padding(16.dp),
                style = MaterialTheme.typography.bodySmall
            )
            return@Box
        }
        Canvas(Modifier.fillMaxWidth().height(220.dp).padding(8.dp)) {
            val allNorth = viewPoints.map { it.north }
            val allEast = viewPoints.map { it.east }
            val minN = allNorth.min(); val maxN = allNorth.max()
            val minE = allEast.min(); val maxE = allEast.max()
            // Square, padded extent so the track never touches the edge and a
            // near-straight line (common right after a session starts) does not
            // get stretched into a misleadingly steep-looking path.
            val span = max(max(maxN - minN, maxE - minE), 5.0) * 1.2
            val centerN = (minN + maxN) / 2.0
            val centerE = (minE + maxE) / 2.0
            val scale = kotlin.math.min(size.width, size.height) / span.toFloat()

            // North is up, east is right, which is a 90 degree rotation from the
            // screen's natural (x right, y down) axes plus a y-flip for "up".
            fun project(north: Double, east: Double): Offset {
                val x = size.width / 2f + ((east - centerE) * scale).toFloat()
                val y = size.height / 2f - ((north - centerN) * scale).toFloat()
                return Offset(x, y)
            }

            fun drawTrack(kind: TrackKind, color: Color, dashed: Boolean) {
                val track = points.filter { it.kind == kind }
                if (track.size < 2) return
                val effect = if (dashed) PathEffect.dashPathEffect(floatArrayOf(10f, 8f)) else null
                for (i in 1 until track.size) {
                    drawLine(
                        color = color,
                        start = project(track[i - 1].north, track[i - 1].east),
                        end = project(track[i].north, track[i].east),
                        strokeWidth = 5f,
                        cap = StrokeCap.Round,
                        pathEffect = effect
                    )
                }
                val head = project(track.last().north, track.last().east)
                drawCircle(color, radius = 8f, center = head)
            }

            // The road goes first so every track sits on top of it.
            if (road != null && road.size >= 2) {
                val w = size.width
                val h = size.height
                var prev = project(road[0][0], road[0][1])
                for (i in 1 until road.size) {
                    val cur = project(road[i][0], road[i][1])
                    val offscreen = (prev.x < 0f && cur.x < 0f) || (prev.x > w && cur.x > w) ||
                        (prev.y < 0f && cur.y < 0f) || (prev.y > h && cur.y > h)
                    if (!offscreen) {
                        drawLine(color = roadColor, start = prev, end = cur, strokeWidth = 16f, cap = StrokeCap.Round)
                    }
                    prev = cur
                }
            }

            // Draw order matters only for overlap legibility, not meaning: coast
            // first (it is expected to diverge most and should not hide the others).
            drawTrack(TrackKind.COAST, coastColor, dashed = true)
            drawTrack(TrackKind.CORRIDOR, corridorColor, dashed = false)
            drawTrack(TrackKind.FUSED, fusedColor, dashed = false)
            drawTrack(TrackKind.TRUTH, truthColor, dashed = false)
        }
        Box(Modifier.align(Alignment.TopStart).padding(8.dp)) {
            Legend(
                truth = if (TrackKind.TRUTH in present) truthColor else null,
                fused = if (TrackKind.FUSED in present) fusedColor else null,
                coast = if (TrackKind.COAST in present) coastColor else null,
                corridor = if (TrackKind.CORRIDOR in present) corridorColor else null,
                road = if (hasRoad) roadColor else null
            )
        }
    }
}

@Composable
private fun Legend(truth: Color?, fused: Color?, coast: Color?, corridor: Color?, road: Color?) {
    Column {
        if (truth != null) LegendRow(truth, "GNSS truth (withheld during blackout, still plotted)")
        if (fused != null) LegendRow(fused, "Fused (UKF)")
        if (coast != null) LegendRow(coast, "Coast baseline (no correction)")
        if (corridor != null) LegendRow(corridor, "Corridor (road-snapped)")
        if (road != null) LegendRow(road, "Locked OSM road")
    }
}

@Composable
private fun LegendRow(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).background(color))
        Spacer(Modifier.width(4.dp))
        Text(label, style = MaterialTheme.typography.labelSmall)
    }
}
