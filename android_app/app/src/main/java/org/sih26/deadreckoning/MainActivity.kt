package org.sih26.deadreckoning

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.provider.OpenableColumns
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Navigation
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import org.sih26.deadreckoning.fusion.FusionSnapshot
import org.sih26.deadreckoning.fusion.PipelinePhase
import org.sih26.deadreckoning.fusion.RoadNetworkState
import org.sih26.deadreckoning.fusion.RoadNetworkStatus
import org.sih26.deadreckoning.fusion.Stage12SpeedModel
import org.sih26.deadreckoning.replay.ReplayLoader
import org.sih26.deadreckoning.replay.ReplaySession
import org.sih26.deadreckoning.sensors.MotionSpeedNetOnnx
import org.sih26.deadreckoning.sensors.SessionRecordingService
import org.sih26.deadreckoning.sessions.SessionRecord
import org.sih26.deadreckoning.sessions.SessionStore
import org.sih26.deadreckoning.ui.ReplayScreen
import org.sih26.deadreckoning.ui.TrackKind
import org.sih26.deadreckoning.ui.TrackPoint
import org.sih26.deadreckoning.ui.TrajectoryCanvas
import java.io.File
import java.io.IOException
import java.util.Locale
import kotlinx.coroutines.launch

/**
 * Four destinations, matching how a field test actually happens: watch the live
 * pipeline while driving, review what got recorded afterwards (as a list, or played
 * back on the trajectory view), and dig into raw diagnostics only when something
 * looks wrong. Keeping diagnostics off the main
 * screen was an explicit call: the person operating the phone during a demo needs
 * four or five numbers, not a debug console.
 *
 * Reached from the navigation drawer, which every destination opens from the same
 * top-bar menu button. [title] is what the top bar shows; [label] is the drawer entry.
 */
private enum class Tab(val label: String, val title: String, val icon: ImageVector) {
    LIVE("Live", "SIH26 Field Test", Icons.Default.Navigation),
    SESSIONS("Sessions", "Recorded sessions", Icons.Default.List),
    REPLAY("Replay", "Replay", Icons.Default.PlayArrow),
    DIAGNOSTICS("Diagnostics", "Diagnostics", Icons.Default.Info)
}

class MainActivity : ComponentActivity() {
    // Read reactively from Compose so the UI updates the moment permission state
    // changes, whether that's the in-app request dialog resolving or the person
    // granting it from system Settings and coming back (onResume, not onCreate,
    // catches that second path - see hasLocationPermission()).
    private var hasLocationPermission by mutableStateOf(false)

    // Advisory, not gating (see the manifest comment on the permission this backs)
    // - a session records fine without it on stock Android. Also re-checked in
    // onResume, since the exemption is granted from a system settings screen this
    // activity does not get a direct callback from.
    private var ignoringBatteryOptimizations by mutableStateOf(false)

    private val permissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        hasLocationPermission = hasLocationPermission()
    }

    // The file the person picked for the Replay page. The system document picker needs
    // no storage permission; this holds the result until the Replay screen has taken it
    // (see consumePickedReplayUri) so a rotation or tab switch cannot re-trigger a load.
    private var pickedReplayUri by mutableStateOf<Uri?>(null)

    private val replayPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        pickedReplayUri = uri
    }

    // Loaded once, lazily, from the same asset SessionRecordingService's live
    // pipeline uses (see MotionSpeedNetOnnx's doc) - null (asset missing/corrupt)
    // degrades a replay import to no Stage 12 track, same as it degrades the live
    // filter to Channel P only.
    private val stage12Model by lazy { MotionSpeedNetOnnx.loadFromAssets(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hasLocationPermission = hasLocationPermission()
        ignoringBatteryOptimizations = isIgnoringBatteryOptimizations()
        val required = buildList {
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
        }.toTypedArray()
        if (required.any { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) {
            permissions.launch(required)
        }
        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) {
                    App(
                        hasLocationPermission = hasLocationPermission,
                        requestPermission = { permissions.launch(required) },
                        openAppSettings = ::openAppSettings,
                        freeStorageBytes = { getExternalFilesDir(null)?.freeSpace ?: Long.MAX_VALUE },
                        ignoringBatteryOptimizations = ignoringBatteryOptimizations,
                        requestIgnoreBatteryOptimizations = ::requestIgnoreBatteryOptimizations,
                        start = { ContextCompat.startForegroundService(this, intent(SessionRecordingService.ACTION_START)) },
                        stop = { startService(intent(SessionRecordingService.ACTION_STOP)) },
                        blackout = { active -> startService(intent(SessionRecordingService.ACTION_SET_BLACKOUT).putExtra(SessionRecordingService.EXTRA_BLACKOUT_ACTIVE, active)) },
                        listSessions = { SessionStore.open(this).recent() },
                        deleteSession = { SessionStore.open(this).delete(it) },
                        export = ::export,
                        pickReplayFile = { replayPicker.launch(arrayOf("*/*")) },
                        pickedReplayUri = pickedReplayUri,
                        consumePickedReplayUri = { pickedReplayUri = null },
                        loadReplayFromUri = ::loadReplay,
                        stage12Model = stage12Model
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Catches granting the permission from the system Settings screen, which
        // resumes this activity without going through the permissions launcher's
        // own callback above.
        hasLocationPermission = hasLocationPermission()
        ignoringBatteryOptimizations = isIgnoringBatteryOptimizations()
    }

    private fun hasLocationPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

    private fun isIgnoringBatteryOptimizations(): Boolean {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        return powerManager.isIgnoringBatteryOptimizations(packageName)
    }

    private fun requestIgnoreBatteryOptimizations() {
        // No ACTION_APPLICATION_DETAILS_SETTINGS fallback needed here the way
        // openAppSettings has one for location: unlike a runtime permission,
        // this request intent does not silently stop presenting itself after a
        // prior denial, so re-showing it is always the right action.
        startActivity(
            Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName"))
        )
    }

    private fun openAppSettings() {
        startActivity(
            Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null))
        )
    }

    /** Runs on a background thread (see ReplayScreen), so blocking I/O is fine here. */
    private fun loadReplay(uri: Uri): ReplaySession {
        val stream = contentResolver.openInputStream(uri) ?: throw IOException("Could not open the selected file.")
        return stream.use { ReplayLoader.load(it, displayName(uri), stage12Model) }
    }

    private fun displayName(uri: Uri): String =
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: uri.lastPathSegment ?: "Selected file"

    private fun intent(action: String) = Intent(this, SessionRecordingService::class.java).setAction(action)

    private fun export(file: File) {
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        startActivity(
            Intent.createChooser(
                Intent(Intent.ACTION_SEND).apply {
                    type = "application/x-ndjson"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                },
                "Export raw SIH26 session"
            )
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun App(
    hasLocationPermission: Boolean,
    requestPermission: () -> Unit,
    openAppSettings: () -> Unit,
    freeStorageBytes: () -> Long,
    ignoringBatteryOptimizations: Boolean,
    requestIgnoreBatteryOptimizations: () -> Unit,
    start: () -> Unit,
    stop: () -> Unit,
    blackout: (Boolean) -> Unit,
    listSessions: () -> List<SessionRecord>,
    deleteSession: (SessionRecord) -> Unit,
    export: (File) -> Unit,
    pickReplayFile: () -> Unit,
    pickedReplayUri: Uri?,
    consumePickedReplayUri: () -> Unit,
    loadReplayFromUri: (Uri) -> ReplaySession,
    stage12Model: Stage12SpeedModel? = null
) {
    var tab by remember { mutableStateOf(Tab.LIVE) }
    // Held here rather than inside the Replay screen so a loaded drive is still there
    // after visiting another destination from the drawer.
    var replaySession by remember { mutableStateOf<ReplaySession?>(null) }
    var telemetry by remember { mutableStateOf<SessionRecordingService.RecordingTelemetry?>(null) }
    var frontEndStatus by remember { mutableStateOf("") }
    val trail = remember { mutableStateListOf<TrackPoint>() }
    // The OSM road the corridor filter last locked onto. Kept after the blackout ends
    // (the snapshot's own copy goes null then) so the drive can still be reviewed
    // against the road it was matched to; cleared when a new recording starts.
    var road by remember { mutableStateOf<List<DoubleArray>?>(null) }
    var records by remember { mutableStateOf(listSessions()) }

    DisposableEffect(Unit) {
        SessionRecordingService.telemetryListener = { t ->
            telemetry = t
            val snap = t.snapshot
            if (t.recording && snap != null) {
                trail.add(TrackPoint(snap.lastTruthNorth, snap.lastTruthEast, TrackKind.TRUTH))
                trail.add(TrackPoint(snap.fusedNorth, snap.fusedEast, TrackKind.FUSED))
                trail.add(TrackPoint(snap.coastNorth, snap.coastEast, TrackKind.COAST))
                // Only present while the corridor filter has a road locked on.
                val matchedN = snap.corridorMatchedNorth
                val matchedE = snap.corridorMatchedEast
                if (matchedN != null && matchedE != null) {
                    trail.add(TrackPoint(matchedN, matchedE, TrackKind.CORRIDOR))
                }
                snap.corridorRoad?.let { if (road !== it) road = it }
                // Cap the trail so a long drive does not grow the canvas's per-frame
                // work without bound: 12000 points is ~10 minutes of history at 5 Hz
                // telemetry with all four tracks present, oldest dropped first. Point
                // count per sample varies now (the corridor track comes and goes), so
                // this trims by total size rather than in fixed groups of three.
                while (trail.size > 12000) trail.removeAt(0)
            }
        }
        SessionRecordingService.statusListener = { frontEndStatus = it }
        onDispose {
            SessionRecordingService.telemetryListener = null
            SessionRecordingService.statusListener = null
        }
    }

    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    fun navigate(target: Tab) {
        tab = target
        if (target == Tab.SESSIONS || target == Tab.REPLAY) records = listSessions()
        scope.launch { drawerState.close() }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet {
                Text(
                    "SIH26 Dead Reckoning",
                    Modifier.padding(horizontal = 28.dp, vertical = 20.dp),
                    style = MaterialTheme.typography.titleMedium
                )
                HorizontalDivider()
                Spacer(Modifier.height(8.dp))
                Tab.values().forEach { entry ->
                    NavigationDrawerItem(
                        label = { Text(entry.label) },
                        icon = { Icon(entry.icon, contentDescription = null) },
                        selected = tab == entry,
                        onClick = { navigate(entry) },
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                    )
                }
            }
        }
    ) {
        Scaffold(topBar = {
            TopAppBar(
                title = { Text(tab.title) },
                navigationIcon = {
                    IconButton(onClick = { scope.launch { drawerState.open() } }) {
                        Icon(Icons.Default.Menu, contentDescription = "Open navigation")
                    }
                }
            )
        }) { padding ->
            Box(Modifier.padding(padding)) {
                when (tab) {
                    Tab.LIVE -> LiveScreen(
                        t = telemetry, trail = trail, road = road,
                        hasLocationPermission = hasLocationPermission,
                        requestPermission = requestPermission,
                        openAppSettings = openAppSettings,
                        freeStorageBytes = freeStorageBytes,
                        ignoringBatteryOptimizations = ignoringBatteryOptimizations,
                        requestIgnoreBatteryOptimizations = requestIgnoreBatteryOptimizations,
                        start = { trail.clear(); road = null; start() }, stop = stop, blackout = blackout
                    )
                    Tab.SESSIONS -> SessionsScreen(records, export) { record ->
                        deleteSession(record)
                        records = listSessions()
                    }
                    Tab.REPLAY -> ReplayScreen(
                        records = records,
                        session = replaySession,
                        onSession = { replaySession = it },
                        pickFile = pickReplayFile,
                        pickedUri = pickedReplayUri,
                        consumePickedUri = consumePickedReplayUri,
                        loadFromUri = loadReplayFromUri,
                        stage12Model = stage12Model
                    )
                    Tab.DIAGNOSTICS -> DiagnosticsScreen(telemetry, frontEndStatus)
                }
            }
        }
    }
}

/** Below this much free space on the session storage volume, a recording is
 * refused outright rather than started only to run out mid-drive: JSONL logging
 * is continuous for the whole session and a multi-hour drive at 100 Hz is
 * megabytes, not kilobytes. Chosen generously below any single realistic session
 * size rather than tuned to a measured rate, since refusing a demo recording
 * over a false positive is worse than the disk actually filling. */
private const val MIN_FREE_STORAGE_BYTES = 50L * 1024 * 1024

@Composable
private fun LiveScreen(
    t: SessionRecordingService.RecordingTelemetry?,
    trail: List<TrackPoint>,
    road: List<DoubleArray>?,
    hasLocationPermission: Boolean,
    requestPermission: () -> Unit,
    openAppSettings: () -> Unit,
    freeStorageBytes: () -> Long,
    ignoringBatteryOptimizations: Boolean,
    requestIgnoreBatteryOptimizations: () -> Unit,
    start: () -> Unit,
    stop: () -> Unit,
    blackout: (Boolean) -> Unit
) {
    val recording = t?.recording == true
    var lowStorageWarningMb by remember { mutableStateOf<Long?>(null) }

    // Scrollable: the velocity and road-matching cards below the canvas take this
    // screen past one phone height while recording.
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        if (!hasLocationPermission) {
            PermissionGate(requestPermission, openAppSettings)
            return@Column
        }

        Button(
            onClick = {
                if (recording) {
                    stop()
                } else {
                    val freeMb = freeStorageBytes() / (1024 * 1024)
                    if (freeStorageBytes() < MIN_FREE_STORAGE_BYTES) lowStorageWarningMb = freeMb else start()
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (recording) "Stop recording" else "Start recording")
        }

        lowStorageWarningMb?.let { freeMb ->
            AlertDialog(
                onDismissRequest = { lowStorageWarningMb = null },
                title = { Text("Low storage") },
                text = { Text("Only ${freeMb} MB free. A field-test recording logs continuously for the whole drive - free up space first rather than risk it running out mid-session.") },
                confirmButton = { TextButton({ lowStorageWarningMb = null }) { Text("OK") } }
            )
        }

        if (!recording) {
            Text(
                "Ready. Start parked so leveling can find gravity, then drive until " +
                    "GNSS heading initialises the filter.",
                style = MaterialTheme.typography.bodyMedium
            )
            if (!ignoringBatteryOptimizations) {
                BatteryOptimizationBanner(requestIgnoreBatteryOptimizations)
            }
            return@Column
        }

        val satelliteSuffix = if (t?.phase != PipelinePhase.RUNNING) {
            satelliteStatusSuffix(t?.gnssSatellitesInView, t?.gnssSatellitesUsed)
        } else ""
        val phase = when {
            t?.blackout == true -> "GNSS BLACKOUT"
            t?.gpsProviderEnabled == false -> "GPS IS OFF"
            t?.waitingForMovement == true -> "WAITING FOR MOVEMENT$satelliteSuffix"
            else -> (t?.phase?.name?.replace('_', ' ') ?: "STARTING") + satelliteSuffix
        }
        PhaseCard(phase = phase, blackout = t?.blackout == true || t?.gpsProviderEnabled == false)

        if (t?.gpsProviderEnabled != false && t?.gnssFixes == 0L && (t?.elapsedS ?: 0.0) > 20.0) {
            val sats = t?.gnssSatellitesInView
            ErrorBanner(
                if (sats == null) {
                    "No GNSS fix and no satellite data yet after ${duration(t?.elapsedS ?: 0.0)}. " +
                        "If this persists, the GNSS chip may not be reporting status on this device."
                } else if (sats == 0) {
                    "No satellites in view after ${duration(t?.elapsedS ?: 0.0)}. The antenna likely has " +
                        "no sky view - move outdoors or away from tall structures/vehicle roofs."
                } else {
                    "$sats satellite(s) in view, ${t?.gnssSatellitesUsed ?: 0} used, but no fix yet after " +
                        "${duration(t?.elapsedS ?: 0.0)}. Weak signal or still resolving - keep waiting outdoors."
                }
            )
        }

        if (t?.gpsProviderEnabled == false) {
            ErrorBanner("GPS is turned off in system settings. Enable location services - this looks identical to \"waiting for a fix\" otherwise.")
        }
        if (t?.loggerFailed == true) {
            ErrorBanner("Recording is NOT being saved: the storage write failed (disk full or unmounted). Stop and check free space.")
        }
        if (t?.fusionFailed == true) {
            ErrorBanner("The on-device fused track has stopped updating after an internal error. Raw sensor/GNSS logging is unaffected and this drive is still worth keeping.")
        }

        TrajectoryCanvas(
            trail, Modifier.fillMaxWidth(), road,
            originLatDeg = t?.originLatDeg, originLonDeg = t?.originLonDeg
        )

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                MetricRow("Speed", "${t?.gnssSpeedMps?.let { fmt(it * 3.6) } ?: "--"} km/h", "Accuracy", "${t?.gnssAccuracyM?.let(::fmt) ?: "--"} m")
                MetricRow("Distance", "${fmt((t?.distanceM ?: 0.0) / 1000)} km", "Elapsed", duration(t?.elapsedS ?: 0.0))
                MetricRow("IMU samples", "${t?.imuSamples ?: 0}", "GNSS fixes", "${t?.gnssFixes ?: 0}")
                MetricRow(
                    "Satellites in view", "${t?.gnssSatellitesInView ?: "--"}",
                    "Satellites used", "${t?.gnssSatellitesUsed ?: "--"}"
                )
                t?.snapshot?.let { snap ->
                    MetricRow("Heading", "${fmt(snap.headingDeg)} deg", "Drift", if (t.blackout) "${fmt(snap.driftMeters)} m (${fmt(snap.driftPercent)}%)" else "n/a - GNSS live")
                }
                if ((t?.droppedSamples ?: 0) > 0) {
                    Text(
                        "WARNING: ${t?.droppedSamples} samples dropped - the writer is falling behind, the raw log has a gap in t.",
                        color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }

        t?.snapshot?.let { snap ->
            VelocityChannelCard(snap, t.blackout)
            RoadMatchingCard(snap, t.blackout)
        }

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Column {
                Text("Software GNSS blackout", style = MaterialTheme.typography.bodyLarge)
                Text("Withholds fixes from the filter only; raw GNSS keeps logging as truth.", style = MaterialTheme.typography.bodySmall)
            }
            Switch(checked = t?.blackout == true, onCheckedChange = blackout)
        }
    }
}

/**
 * Which speed source is feeding the UKF's Channel A slot right now. Mirrors the
 * pipeline's own rule (`finalChannelASpeed = stage12 ?: channelP`): Channel P is
 * applied every cycle it has a value, GNSS live or not (it reseeds from each fix and
 * integrates in between), and Stage 12 takes the slot over only during a blackout
 * once its pre-blackout calibration is trusted. The corridor is not a speed source
 * and lives in [RoadMatchingCard].
 *
 * Only shown while a session is running (the caller gates on t?.snapshot being
 * non-null), so there is nothing to render before the filter initialises.
 */
@Composable
private fun VelocityChannelCard(snap: FusionSnapshot, blackout: Boolean) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Velocity channels", style = MaterialTheme.typography.titleSmall)
            ChannelStatusRow(
                name = "Channel P (physics)",
                valueText = snap.channelPSpeed?.let { "${fmt(it)} m/s" } ?: "not resolved",
                active = snap.channelPSpeed != null && !snap.stage12Active
            )
            ChannelStatusRow(
                name = "Stage 12 (MotionSpeedNet)",
                valueText = when {
                    snap.stage12SpeedMps != null -> "${fmt(snap.stage12SpeedMps)} m/s"
                    blackout -> "warming up / not calibrated"
                    else -> "standby - blackout only"
                },
                active = snap.stage12Active
            )
            Text(
                if (blackout) {
                    if (snap.stage12Active) "Stage 12 is feeding the filter; Channel P is the fallback."
                    else "Channel P is feeding the filter; Stage 12 has no trusted prediction yet."
                } else {
                    "GNSS is live: Channel P feeds the filter (reseeded from each fix). " +
                        "Stage 12 only takes over during a blackout."
                },
                style = MaterialTheme.typography.labelSmall
            )
        }
    }
}

/**
 * The OSM road-matching corridor: whether the road network downloaded, whether a road
 * got locked at blackout start, how well the turn signature matches it, and whether
 * the road-snapped position is actually being applied to the filter.
 */
@Composable
private fun RoadMatchingCard(snap: FusionSnapshot, blackout: Boolean) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Road matching (OSM)", style = MaterialTheme.typography.titleSmall)
            ChannelStatusRow(
                name = "Road network",
                valueText = roadNetworkText(snap.roadNetworkStatus),
                active = snap.roadNetworkStatus.state == RoadNetworkState.READY
            )
            ChannelStatusRow(
                name = "Road lock",
                valueText = when {
                    snap.corridorRoadLocked -> "locked - conf ${fmt(snap.corridorConfidence * 100)}%"
                    blackout -> "not locked (no road in range at blackout start)"
                    else -> "locks when a blackout starts"
                },
                active = snap.corridorRoadLocked
            )
            if (snap.corridorRoadLocked) {
                ChannelStatusRow(
                    name = "Position correction",
                    valueText = if (snap.corridorActive) {
                        "applied" + (snap.corridorPositionCorrectionM?.let { " - ${fmt(it)} m pull" } ?: "")
                    } else {
                        "held back (confidence too low)" +
                            (snap.corridorPositionCorrectionM?.let { " - ${fmt(it)} m off road" } ?: "")
                    },
                    active = snap.corridorActive
                )
            }
        }
    }
}

private fun roadNetworkText(status: RoadNetworkStatus): String = when (status.state) {
    RoadNetworkState.IDLE -> "waiting for first GNSS fix"
    RoadNetworkState.FETCHING -> "downloading roads..."
    RoadNetworkState.RETRYING -> "no data yet - retrying (${status.attempt} failed)"
    RoadNetworkState.READY -> "ready - ${status.segmentCount} segments"
    RoadNetworkState.FAILED -> "unavailable after ${status.attempt} tries - corridor off"
}

@Composable
private fun ChannelStatusRow(name: String, valueText: String, active: Boolean) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Box(
                Modifier.size(8.dp).background(
                    color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant,
                    shape = androidx.compose.foundation.shape.CircleShape
                )
            )
            Text(name, style = MaterialTheme.typography.bodyMedium, fontWeight = if (active) FontWeight.Bold else FontWeight.Normal)
        }
        Text(valueText, style = MaterialTheme.typography.bodyMedium, fontWeight = if (active) FontWeight.Bold else FontWeight.Normal)
    }
}

/**
 * Shown in place of the recording controls when location permission is missing.
 *
 * Before this, tapping "Start recording" without it granted did nothing visible:
 * the service checked the permission internally and stopped itself with only a
 * log-level status message nobody was looking at (`Diagnostics` is deliberately
 * off the main screen). On a borrowed demo device where the permission dialog
 * gets dismissed by accident, that reads as "the app is broken" rather than "grant
 * a permission" - exactly the failure mode a bottom-of-the-stack error message is
 * built to avoid. [requestPermission] re-shows the system dialog for a first
 * denial; once the OS considers it permanently denied, that dialog silently
 * no-ops instead of reappearing, which is what [openAppSettings] is for.
 */
/**
 * Advisory-only, shown before a recording starts. The foreground service plus
 * its partial wake lock already keep stock Android from throttling sensor
 * delivery with the screen off (see `SessionRecordingService`'s class docs,
 * point 7) - this is for the layer several OEM skins (MIUI, ColorOS, and
 * similar heavier customizations) add on top of stock Doze, which the wake lock
 * alone does not reliably survive on those. Exempting the app from battery
 * optimization is the one piece of that OEM-specific gap that is fixable from
 * inside the app via a documented API, rather than a "clear the app from
 * recents and disable autostart controls" set of vendor-specific settings
 * screens with no common Android API - flagged as the known remaining gap in
 * Master_Implementation_Plan.md rather than silently left unfixed.
 */
@Composable
private fun BatteryOptimizationBanner(request: () -> Unit) {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                "For a long unattended drive, exempt this app from battery optimization - " +
                    "some phones throttle sensors in the background even with a recording " +
                    "notification showing.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
            OutlinedButton(onClick = request) { Text("Disable battery optimization") }
        }
    }
}

@Composable
private fun PermissionGate(requestPermission: () -> Unit, openAppSettings: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Location permission needed", style = MaterialTheme.typography.titleMedium)
            Text(
                "Recording needs GPS fixes as the drift ground truth. Grant location " +
                    "access to start.",
                style = MaterialTheme.typography.bodyMedium
            )
            Button(onClick = requestPermission, modifier = Modifier.fillMaxWidth()) {
                Text("Grant permission")
            }
            OutlinedButton(onClick = openAppSettings, modifier = Modifier.fillMaxWidth()) {
                Text("Open app settings")
            }
            Text(
                "If the permission dialog stopped appearing, it was likely denied " +
                    "permanently - use \"Open app settings\" and enable location there instead.",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
private fun ErrorBanner(text: String) {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
        Text(text, Modifier.padding(12.dp), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onErrorContainer)
    }
}

@Composable
private fun PhaseCard(phase: String, blackout: Boolean) {
    val container = if (blackout) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.secondaryContainer
    val content = if (blackout) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = container)) {
        Text(phase, Modifier.padding(12.dp), style = MaterialTheme.typography.titleMedium, color = content)
    }
}

@Composable
private fun MetricRow(label1: String, value1: String, label2: String, value2: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Column(Modifier.weight(1f)) { Text(label1, style = MaterialTheme.typography.labelSmall); Text(value1, style = MaterialTheme.typography.bodyLarge) }
        Column(Modifier.weight(1f)) { Text(label2, style = MaterialTheme.typography.labelSmall); Text(value2, style = MaterialTheme.typography.bodyLarge) }
    }
}

@Composable
private fun SessionsScreen(records: List<SessionRecord>, export: (File) -> Unit, delete: (SessionRecord) -> Unit) {
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        if (records.isEmpty()) {
            Text("No completed sessions yet. Recordings appear here after you stop them.")
            return@Column
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(records, key = { it.id }) { r -> SessionCard(r, export, delete) }
        }
    }
}

@Composable
private fun SessionCard(r: SessionRecord, export: (File) -> Unit, delete: (SessionRecord) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    var confirmingDelete by remember { mutableStateOf(false) }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp).clickable { expanded = !expanded }, verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Drive - ${r.startedUtc.take(16).replace('T', ' ')} UTC", style = MaterialTheme.typography.titleMedium)
            Text("${duration(r.durationS)} - ${fmt(r.distanceM / 1000)} km - ${r.imuSamples} IMU samples - ${r.gnssFixes} GNSS fixes")
            if (expanded) {
                Text("Blackout duration: ${duration(r.blackoutDurationS)}")
                Text("Final blackout drift: ${r.finalDriftM?.let { "${fmt(it)} m" } ?: "not measured (no blackout this session)"}")
                Text("Raw file: ${r.rawFile.name}", style = MaterialTheme.typography.bodySmall)
                Text("File on disk: ${r.rawFile.length() / 1024} KB", style = MaterialTheme.typography.bodySmall)
                if (r.droppedSamples > 0) {
                    Text(
                        "${r.droppedSamples} samples dropped during recording - log has a gap in t.",
                        color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall
                    )
                }
                if (r.loggerFailed) {
                    Text(
                        "Storage write failed partway through - this recording is truncated.",
                        color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall
                    )
                }
                if (r.fusionFailed) {
                    Text(
                        "Fusion pipeline failed partway through - raw sensor/GNSS data is intact, fused track is not.",
                        color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button({ export(r.rawFile) }) { Text("Export JSONL") }
                    OutlinedButton({ confirmingDelete = true }) { Text("Delete") }
                }
            }
        }
    }
    if (confirmingDelete) {
        AlertDialog(
            onDismissRequest = { confirmingDelete = false },
            title = { Text("Delete this session?") },
            text = { Text("This removes the raw JSONL and its metadata from the phone permanently. Export first if you need it.") },
            confirmButton = { TextButton({ confirmingDelete = false; delete(r) }) { Text("Delete") } },
            dismissButton = { TextButton({ confirmingDelete = false }) { Text("Cancel") } }
        )
    }
}

@Composable
private fun DiagnosticsScreen(t: SessionRecordingService.RecordingTelemetry?, frontEndStatus: String) {
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            "Raw pipeline internals. Nothing here is smoothed for presentation; it is " +
                "exactly what the fusion front end reports.",
            style = MaterialTheme.typography.bodySmall
        )
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                DiagRow("Recording", if (t?.recording == true) "yes" else "no")
                DiagRow("Pipeline phase", t?.phase?.name ?: "n/a")
                DiagRow("Waiting for moving fix", if (t?.waitingForMovement == true) "yes" else "no")
                DiagRow("Software blackout", if (t?.blackout == true) "ACTIVE" else "off")
                DiagRow("GPS provider (system setting)", if (t?.gpsProviderEnabled == false) "DISABLED" else "enabled")
                DiagRow("Satellites in view / used", "${t?.gnssSatellitesInView ?: "no data yet"} / ${t?.gnssSatellitesUsed ?: "no data yet"}")
                DiagRow("Dropped samples (writer behind)", "${t?.droppedSamples ?: 0}")
                DiagRow("Storage write failed", if (t?.loggerFailed == true) "YES - not saving" else "no")
                DiagRow("Fusion pipeline failed", if (t?.fusionFailed == true) "YES - raw-only" else "no")
                t?.snapshot?.let { snap ->
                    DiagRow("IMU sample rate", "${fmt(snap.imuRateHz)} Hz")
                    DiagRow("Filter step latency", "${fmt(snap.stepLatencyMs)} ms")
                    DiagRow("ZUPT active this cycle", if (snap.zuptActive) "yes" else "no")
                    DiagRow("Channel P speed", snap.channelPSpeed?.let { "${fmt(it)} m/s" } ?: "not resolved")
                    DiagRow("Stage 12 speed", snap.stage12SpeedMps?.let { "${fmt(it)} m/s" } ?: "not resolved")
                    DiagRow("Stage 12 feeding UKF this cycle", if (snap.stage12Active) "yes" else "no")
                    DiagRow("Road network (OSM)", roadNetworkText(snap.roadNetworkStatus))
                    DiagRow("Corridor road locked", if (snap.corridorRoadLocked) "yes" else "no")
                    DiagRow("Corridor progress confidence", "${fmt(snap.corridorConfidence * 100)}%")
                    DiagRow("Corridor feeding UKF this cycle", if (snap.corridorActive) "yes" else "no")
                    DiagRow(
                        "Corridor matched position (N,E)",
                        if (snap.corridorMatchedNorth != null && snap.corridorMatchedEast != null) {
                            "${fmt(snap.corridorMatchedNorth)}, ${fmt(snap.corridorMatchedEast)} m"
                        } else "n/a"
                    )
                    DiagRow("Fused position (N,E)", "${fmt(snap.fusedNorth)}, ${fmt(snap.fusedEast)} m")
                    DiagRow("Coast position (N,E)", "${fmt(snap.coastNorth)}, ${fmt(snap.coastEast)} m")
                    DiagRow("Truth position (N,E)", "${fmt(snap.lastTruthNorth)}, ${fmt(snap.lastTruthEast)} m")
                }
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Front-end status", style = MaterialTheme.typography.titleSmall)
                Text(frontEndStatus.ifBlank { "No GNSS-triggered update yet this session." }, style = MaterialTheme.typography.bodySmall)
            }
        }
        Text(
            "ACCEL / GYRO / MAG are raw android.hardware sensors, never fused/virtual " +
                "ones. GPS is LocationManager.GPS_PROVIDER, never FusedLocationProviderClient.",
            style = MaterialTheme.typography.labelSmall
        )
    }
}

@Composable
private fun DiagRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

/** Short parenthetical appended to the phase label while still waiting for a fix,
 * e.g. "WAITING FOR GNSS (0 sats)" vs "WAITING FOR GNSS (7 sats, 2 used)". Empty
 * once running, since satellite count stops being the interesting number once a
 * fix is already flowing. */
private fun satelliteStatusSuffix(satellitesInView: Int?, satellitesUsed: Int?): String {
    if (satellitesInView == null) return ""
    return if (satellitesInView == 0) " (0 sats)" else " ($satellitesInView sats, ${satellitesUsed ?: 0} used)"
}

private fun fmt(value: Double) = String.format(Locale.ROOT, "%.2f", value)
private fun duration(seconds: Double) = "%dm %02ds".format(Locale.ROOT, (seconds / 60).toInt(), (seconds % 60).toInt())
