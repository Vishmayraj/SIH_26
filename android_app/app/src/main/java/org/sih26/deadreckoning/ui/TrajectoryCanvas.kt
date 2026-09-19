package org.sih26.deadreckoning.ui

import android.graphics.Color as AndroidColor
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
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Polyline
import org.sih26.deadreckoning.fusion.LocalFrame

/**
 * One track sample in the local tangent frame (meters, north/east), tagged by
 * source so the tracks the field-test milestone calls out - raw GNSS truth, the fused
 * (UKF) estimate, the no-correction coast baseline, the corridor filter's road-snapped
 * position, and (on a replay import) the Stage 12 model's own dead-reckoned track -
 * stay visually and semantically separate. Nothing here decides what counts as "true"; it just
 * renders whatever FusionSnapshot or ReplaySession already reports.
 */
enum class TrackKind { TRUTH, FUSED, COAST, CORRIDOR, STAGE12 }

data class TrackPoint(val north: Double, val east: Double, val kind: TrackKind)

/**
 * A real, downloaded OpenStreetMap basemap (osmdroid, `TileSourceFactory.MAPNIK`)
 * with every track drawn as a colored overlay on top of it. Replaces the earlier
 * hand-rolled blank-canvas view: this is what "is the fused track actually tracking
 * truth, and is that materially better than doing nothing" now looks like against
 * real streets instead of an unlabeled north-up square.
 *
 * [originLatDeg]/[originLonDeg] is the session's local-frame fix ([LocalFrame]'s
 * lat0/lon0) that [points], [road] and [extentPoints] (all North/East metres) are
 * relative to - without it there is nothing to project onto a real map, so the view
 * falls back to [emptyMessage] exactly as it does with fewer than two points.
 *
 * [road] is the OSM road polyline the corridor filter locked onto (North/East metres,
 * from FusionSnapshot.corridorRoad), drawn as an underlay.
 *
 * [extentPoints], when given, sets the initial camera position from that list instead
 * of from [points]: a replay passes its whole session here so the view is centered on
 * the whole drive rather than re-centering on every frame as [points] grows.
 */
@Composable
fun TrajectoryCanvas(
    points: List<TrackPoint>,
    modifier: Modifier = Modifier,
    road: List<DoubleArray>? = null,
    extentPoints: List<TrackPoint>? = null,
    emptyMessage: String = "Trajectory will appear once the fix is initialised.",
    originLatDeg: Double? = null,
    originLonDeg: Double? = null
) {
    val viewPoints = extentPoints ?: points
    val truthColor = AndroidColor.rgb(0x2E, 0x7D, 0x32)    // GNSS ground truth - green
    val fusedColor = AndroidColor.rgb(0x15, 0x65, 0xC0)    // UKF fused estimate - blue
    val coastColor = AndroidColor.rgb(0xE6, 0x51, 0x00)    // no-correction coast baseline - orange
    val corridorColor = AndroidColor.rgb(0x8E, 0x24, 0xAA) // corridor road-snapped position - purple
    val stage12Color = AndroidColor.rgb(0xC6, 0x28, 0x28)  // Stage 12 model-only replay track - red
    val roadColor = AndroidColor.argb(110, 120, 120, 120)
    val present = viewPoints.mapTo(HashSet()) { it.kind }
    val hasRoad = road != null && road.size >= 2

    Box(modifier.fillMaxWidth().height(220.dp).background(MaterialTheme.colorScheme.surfaceVariant)) {
        if (points.size < 2 || originLatDeg == null || originLonDeg == null) {
            Text(
                emptyMessage,
                Modifier.align(Alignment.Center).padding(16.dp),
                style = MaterialTheme.typography.bodySmall
            )
            return@Box
        }

        val context = LocalContext.current
        val frame = remember(originLatDeg, originLonDeg) { LocalFrame(originLatDeg, originLonDeg) }
        fun geo(north: Double, east: Double): GeoPoint {
            val ll = frame.toLatLon(north, east)
            return GeoPoint(ll[0], ll[1])
        }

        val mapView = remember {
            MapView(context).apply {
                setTileSource(TileSourceFactory.MAPNIK)
                setMultiTouchControls(true)
                controller.setZoom(17.0)
            }
        }

        AndroidView(factory = { mapView }, modifier = Modifier.fillMaxWidth().height(220.dp)) { view ->
            view.overlays.clear()

            // The road goes first so every track sits on top of it.
            if (hasRoad) {
                val line = Polyline(view)
                line.outlinePaint.color = roadColor
                line.outlinePaint.strokeWidth = 14f
                line.setPoints(road!!.map { geo(it[0], it[1]) })
                view.overlays.add(line)
            }

            // Draw order matters only for overlap legibility, not meaning: coast
            // first (it is expected to diverge most and should not hide the others).
            fun drawTrack(kind: TrackKind, color: Int) {
                val track = points.filter { it.kind == kind }
                if (track.size < 2) return
                val line = Polyline(view)
                line.outlinePaint.color = color
                line.outlinePaint.strokeWidth = 7f
                line.setPoints(track.map { geo(it.north, it.east) })
                view.overlays.add(line)
            }
            drawTrack(TrackKind.COAST, coastColor)
            drawTrack(TrackKind.STAGE12, stage12Color)
            drawTrack(TrackKind.CORRIDOR, corridorColor)
            drawTrack(TrackKind.FUSED, fusedColor)
            drawTrack(TrackKind.TRUTH, truthColor)

            if (viewPoints.isNotEmpty()) {
                val centerNorth = viewPoints.sumOf { it.north } / viewPoints.size
                val centerEast = viewPoints.sumOf { it.east } / viewPoints.size
                view.controller.setCenter(geo(centerNorth, centerEast))
            }
            view.invalidate()
        }

        Box(Modifier.align(Alignment.TopStart).padding(8.dp)) {
            Legend(
                truth = if (TrackKind.TRUTH in present) Color(0xFF2E7D32) else null,
                fused = if (TrackKind.FUSED in present) Color(0xFF1565C0) else null,
                coast = if (TrackKind.COAST in present) Color(0xFFE65100) else null,
                corridor = if (TrackKind.CORRIDOR in present) Color(0xFF8E24AA) else null,
                stage12 = if (TrackKind.STAGE12 in present) Color(0xFFC62828) else null,
                road = if (hasRoad) Color(0xFF787878) else null
            )
        }
    }
}

@Composable
private fun Legend(truth: Color?, fused: Color?, coast: Color?, corridor: Color?, stage12: Color?, road: Color?) {
    Column {
        if (truth != null) LegendRow(truth, "GNSS truth (withheld during blackout, still plotted)")
        if (fused != null) LegendRow(fused, "Fused (UKF)")
        if (coast != null) LegendRow(coast, "Coast baseline (no correction)")
        if (corridor != null) LegendRow(corridor, "Corridor (road-snapped)")
        if (stage12 != null) LegendRow(stage12, "Stage 12 model (IMU-only, this import)")
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
