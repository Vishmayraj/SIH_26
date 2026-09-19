package org.sih26.deadreckoning.sensors

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import org.sih26.deadreckoning.MainActivity
import org.sih26.deadreckoning.R
import org.sih26.deadreckoning.fusion.FusionPipeline
import org.sih26.deadreckoning.fusion.FusionSnapshot
import org.sih26.deadreckoning.fusion.PipelinePhase
import org.sih26.deadreckoning.sessions.SessionRecord
import org.sih26.deadreckoning.sessions.SessionStore
import java.io.File

/**
 * Foreground service owning sensor ingest, GNSS, logging and the fusion pipeline.
 *
 * The non-negotiable traps this class exists to not fall into, each stated once
 * here rather than scattered as comments at each call site, because getting any one
 * of them wrong silently invalidates every number this project produces:
 *
 * 1. Raw sensors only: [Sensor.TYPE_ACCELEROMETER] and [Sensor.TYPE_GYROSCOPE], never
 *    `TYPE_LINEAR_ACCELERATION` or `TYPE_ROTATION_VECTOR`. Those apply an
 *    undocumented vendor fusion that competes with, and would circularly validate,
 *    this project's own filter.
 * 2. Raw GNSS only: [LocationManager.GPS_PROVIDER], never
 *    `FusedLocationProviderClient`, which already blends IMU, WiFi and cell into its
 *    position.
 * 3. One clock: every timestamp derives from [SensorEvent.timestamp] /
 *    [Location.getElapsedRealtimeNanos], the monotonic elapsedRealtime base, never
 *    wall time, or the two streams cannot be aligned afterwards.
 * 4. Sensor callbacks run on a dedicated [HandlerThread], never the main thread, and
 *    the whole fusion step happens there too - [FusionPipeline] is not
 *    thread-safe by design (see its own docs) and is owned by this one thread.
 * 5. A persistent foreground notification, or Android throttles or kills sensor
 *    delivery once the screen turns off, which is exactly when a test drive is
 *    unattended.
 * 6. The blackout is simulated in software. GNSS keeps arriving and keeps being
 *    logged the entire time; only whether it reaches the filter is gated. Airplane
 *    mode would destroy the withheld ground truth and cost 10-30s of reacquisition.
 * 7. A foreground notification keeps the process alive, but several OEM battery
 *    managers still throttle sensor *delivery rate* under Doze once the screen is
 *    off, independent of whether the process is killed. A partial [PowerManager.WakeLock]
 *    is held for the duration of the recording (see [wakeLock]) so a real unattended
 *    drive with the screen off is not silently sampled at a lower rate than the one
 *    this project's numbers were measured against.
 * 8. Two failure modes must not be allowed to silently end a recording:
 *    [SessionLogger] hitting an unrecoverable disk-write error, and
 *    [FusionPipeline.onImu] throwing on a numerical edge case. Both are caught at
 *    their call site, surfaced to the UI ([RecordingTelemetry.loggerFailed] /
 *    [RecordingTelemetry.fusionFailed]), and degrade rather than crash: a logger
 *    failure stops further writes but the app keeps running so the driver notices,
 *    and a fusion failure disables the fused track for the rest of the session
 *    while raw sensor/GNSS logging - the one asset a field test cannot repeat -
 *    continues uncorrected.
 */
class SessionRecordingService : Service() {

    companion object {
        const val NOTIFICATION_CHANNEL_ID = "recording"
        const val NOTIFICATION_ID = 1

        const val ACTION_START = "org.sih26.deadreckoning.action.START"
        const val ACTION_STOP = "org.sih26.deadreckoning.action.STOP"
        const val ACTION_SET_BLACKOUT = "org.sih26.deadreckoning.action.SET_BLACKOUT"
        const val EXTRA_BLACKOUT_ACTIVE = "blackout_active"

        // 100 Hz. SENSOR_DELAY_FASTEST is deliberately NOT used: on some devices it
        // delivers 400-500 Hz, burning battery and CPU for resolution nothing here
        // uses. An explicit period pins the rate to what was actually measured and
        // designed against.
        const val SAMPLING_PERIOD_US = 10_000

        const val GNSS_MIN_INTERVAL_MS = 1000L
        const val GNSS_MIN_DISTANCE_M = 0f

        /** Callback for the UI. Set by MainActivity while bound/visible; the service
         * runs correctly with this null, since logging and fusion do not depend on
         * anyone watching. */
        @Volatile
        var snapshotListener: ((FusionSnapshot?) -> Unit)? = null

        @Volatile
        var statusListener: ((String) -> Unit)? = null

        @Volatile
        var telemetryListener: ((RecordingTelemetry) -> Unit)? = null
    }

    /** UI-facing diagnostics; it mirrors pipeline state instead of inventing a
     * second lifecycle state machine. */
    data class RecordingTelemetry(
        val recording: Boolean,
        val phase: PipelinePhase?,
        val waitingForMovement: Boolean,
        val blackout: Boolean,
        val elapsedS: Double,
        val distanceM: Double,
        val imuSamples: Long,
        val gnssFixes: Long,
        val gnssSpeedMps: Double?,
        val gnssAccuracyM: Double?,
        val snapshot: FusionSnapshot?,
        /** Samples the logger's bounded queue had to drop because the writer
         * thread fell behind. Non-zero means the raw log has a real gap in `t`
         * somewhere - worth knowing about during the drive, not just after
         * replaying the log, since it is otherwise silent. */
        val droppedSamples: Long,
        /** True once [SessionLogger] has hit an unrecoverable write error (disk
         * full, storage unmounted). Distinct from [droppedSamples]: this means
         * nothing further is being saved at all, not just that the writer fell
         * behind - a driver needs to know this immediately, not after the drive. */
        val loggerFailed: Boolean,
        /** True once the fusion pipeline has thrown on a numerical edge case
         * (see [maybeEmitImuSample]) and been disabled for the rest of the
         * session. Raw accelerometer/gyroscope/GNSS logging is unaffected - only
         * the on-device fused track and live drift readout stop updating. */
        val fusionFailed: Boolean,
        /** [LocationManager.GPS_PROVIDER]'s enabled/disabled toggle in system
         * settings, checked fresh on every publish. Permission being granted does
         * not mean GPS is actually on - a phone with the setting off looks
         * identical to a phone still waiting for its first fix (both sit at
         * WAITING_FOR_GNSS with zero GNSS fixes), which is a real field-test trap:
         * without this, the only way to tell them apart is to notice fixes never
         * arrive at all. */
        val gpsProviderEnabled: Boolean,
        /** Satellites currently in view of the GNSS chip, from
         * [android.location.GnssStatus.getSatelliteCount], updated independently of
         * whether a fix has ever been produced. This is the diagnostic a zero-fix
         * session cannot otherwise give you: it separates "0 satellites in view"
         * (antenna has no sky view - indoors, underground, deep urban canyon; no fix
         * is possible no matter how long this runs) from "N in view, 0 used" (weak
         * signal / multipath / still resolving ephemeris - a fix may still arrive)
         * from "N in view, M used, still no Location callback" (a genuine
         * registration or platform bug worth escalating, since the chip has
         * everything it needs). Null until the first status update arrives - a real
         * "no data yet" distinct from a real zero. */
        val gnssSatellitesInView: Int?,
        /** Of [gnssSatellitesInView], how many the chip is actually using toward a
         * fix this update. Null alongside [gnssSatellitesInView]. */
        val gnssSatellitesUsed: Int?,
        /** [FusionPipeline.localFrame]'s origin fix, mirrored here so the Live
         * screen's map (TrajectoryCanvas) can project North/East metres onto real
         * lat/lon. Null until the pipeline has its first GNSS fix. */
        val originLatDeg: Double?,
        val originLonDeg: Double?
    )

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private lateinit var thread: HandlerThread
    private lateinit var handler: Handler

    /** Held only between [startRecording] and [stopRecording]/[onDestroy], never
     * acquired at [onCreate] time - a service that exists but is not recording has
     * no reason to keep the CPU awake. Timed out generously (max session length a
     * field test would plausibly run unattended) as a backstop against a code path
     * that fails to release it; a genuinely longer recording simply reacquires
     * nothing since the lock is already held and Android does not double-count. */
    private var wakeLock: PowerManager.WakeLock? = null

    private var logger: SessionLogger? = null
    private var pipeline: FusionPipeline? = null

    /** Held so [stopRecording]/[onDestroy] can cancel a pending road-network retry and
     * release its thread; the pipeline only sees it through the RoadNetworkProvider
     * interface. */
    private var roadProvider: OverpassRoadNetworkProvider? = null
    private lateinit var sessionStore: SessionStore
    private var sessionId: String? = null
    private var sessionStartedUtc: String = ""
    private var gnssFixes: Long = 0
    private var distanceM: Double = 0.0
    private var lastGnssTimestampNs: Long? = null
    private var lastGnssSpeedMps: Double? = null
    private var lastGnssAccuracyM: Double? = null
    private var blackoutStartedNs: Long? = null
    private var blackoutDurationS: Double = 0.0
    private var finalBlackoutDriftM: Double? = null
    private var lastSnapshot: FusionSnapshot? = null
    private var lastTelemetryPublishNs: Long = 0L
    private var gnssSatellitesInView: Int? = null
    private var gnssSatellitesUsed: Int? = null

    /**
     * Satellite-level visibility, independent of whether a fix has ever been
     * produced. [LocationListener] only fires on a completed fix, so a phone that
     * never gets one (indoors, deep urban canyon, no AGPS data yet) looks
     * completely silent on that path alone - this is the callback that actually
     * runs while "waiting for GNSS" is stuck, and the only way to tell "the chip
     * sees nothing" from "the chip sees plenty but can't resolve a fix yet" from
     * the field. Registered alongside [locationListener] in [startRecording],
     * unregistered in [stopRecording]/[onDestroy].
     */
    private val gnssStatusCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(status: GnssStatus) {
            val total = status.satelliteCount
            var used = 0
            for (i in 0 until total) if (status.usedInFix(i)) used++
            gnssSatellitesInView = total
            gnssSatellitesUsed = used
            publishTelemetry()
        }

        override fun onStopped() {
            gnssSatellitesInView = null
            gnssSatellitesUsed = null
        }
    }

    private var sessionStartElapsedNs: Long = 0L
    private var latestMagnetometer: FloatArray? = null

    /** Set once [FusionPipeline.onImu] throws. Raw sensor logging is on the
     * critical path and must survive a bug in the newer, less-battle-tested
     * fusion math; see [maybeEmitImuSample]. */
    private var fusionFailed = false

    private val sensorListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            when (event.sensor.type) {
                Sensor.TYPE_MAGNETIC_FIELD -> {
                    // Magnetometer arrives on its own cadence and nothing here fuses
                    // it (see session.py's schema docs on why it is logged anyway);
                    // the latest sample is attached to the next IMU record rather
                    // than logged as its own event, since the schema ties mx/my/mz
                    // to the imu record.
                    latestMagnetometer = event.values.copyOf()
                    return
                }
                Sensor.TYPE_ACCELEROMETER -> {
                    lastAccel = event.values.copyOf()
                    // Emit on the accel clock only. Calling this after a gyro event
                    // logged the same accel sample twice (with a stale timestamp),
                    // roughly doubling the record rate and corrupting dt.
                    maybeEmitImuSample(event.timestamp)
                }
                Sensor.TYPE_GYROSCOPE -> {
                    lastGyro = event.values.copyOf()
                }
                else -> return
            }
        }

        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit
    }

    // Accelerometer and gyroscope are delivered as separate events even when
    // requested at the same rate; a cycle is emitted whenever a fresh accelerometer
    // sample arrives, carrying whatever the most recent gyro sample was. This
    // matches how the offline reference already treats one IMU record as one
    // (accel, gyro) pair and avoids either stream blocking on the other.
    private var lastAccel: FloatArray? = null
    private var lastGyro: FloatArray? = null

    private fun maybeEmitImuSample(timestampNs: Long) {
        val accel = lastAccel ?: return
        val gyro = lastGyro ?: return

        val tSeconds = (timestampNs - sessionStartElapsedNs) / 1_000_000_000.0
        val mag = latestMagnetometer

        logger?.logImu(
            tSeconds,
            accel[0].toDouble(), accel[1].toDouble(), accel[2].toDouble(),
            gyro[0].toDouble(), gyro[1].toDouble(), gyro[2].toDouble(),
            mag?.get(0)?.toDouble(), mag?.get(1)?.toDouble(), mag?.get(2)?.toDouble()
        )

        // The raw imu/gnss records above are already queued for the writer by this
        // point regardless of what happens next - that is the one asset a field
        // test cannot afford to lose. The fusion step is comparatively young code
        // (a fresh Kotlin port, exercised by the parity/front-end gates but not
        // yet a real drive) and can throw on a genuine numerical edge case (an
        // ill-conditioned covariance, a singular matrix - see LinAlg.kt). One bad
        // cycle must not take the whole recording down with it: catch, disable
        // fusion for the remainder of this session so it does not throw again
        // every cycle at 100 Hz, and keep the raw log running uncorrected.
        val snapshot = if (fusionFailed) null else try {
            pipeline?.onImu(
                timestampNs,
                doubleArrayOf(accel[0].toDouble(), accel[1].toDouble(), accel[2].toDouble()),
                doubleArrayOf(gyro[0].toDouble(), gyro[1].toDouble(), gyro[2].toDouble())
            )
        } catch (t: Throwable) {
            fusionFailed = true
            publishStatus(
                "ERROR: fusion pipeline failed (${t.javaClass.simpleName}: ${t.message}); " +
                    "raw IMU/GNSS logging continues uncorrected for the rest of this session."
            )
            publishTelemetry()
            null
        }
        if (snapshot != null) {
            lastSnapshot = snapshot
            if (blackoutActive) finalBlackoutDriftM = snapshot.driftMeters
            logger?.logFused(snapshot.tSeconds, snapshot.fusedNorth, snapshot.fusedEast)
            logger?.logCoast(snapshot.tSeconds, snapshot.coastNorth, snapshot.coastEast)
            publishSnapshot(snapshot)
            // Compose is a display consumer, not part of the 100 Hz hot path.
            // Publish at 5 Hz while retaining every raw/fused record in JSONL.
            if (timestampNs - lastTelemetryPublishNs >= 200_000_000L) {
                lastTelemetryPublishNs = timestampNs
                publishTelemetry()
            }
        }
    }

    private var blackoutActive = false

    private val locationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            val tSeconds = (location.elapsedRealtimeNanos - sessionStartElapsedNs) / 1_000_000_000.0
            val speed = if (location.hasSpeed()) location.speed.toDouble() else 0.0
            val bearing = if (location.hasBearing()) location.bearing.toDouble() else 0.0
            val accuracy = if (location.hasAccuracy()) location.accuracy.toDouble() else 50.0
            gnssFixes++
            lastGnssSpeedMps = speed
            lastGnssAccuracyM = accuracy
            lastGnssTimestampNs?.let { previous ->
                val dt = (location.elapsedRealtimeNanos - previous) / 1_000_000_000.0
                if (dt in 0.0..10.0) distanceM += speed * dt
            }
            lastGnssTimestampNs = location.elapsedRealtimeNanos

            // Always logged, regardless of blackout state - withheld fixes are the
            // ground truth the drift is measured against.
            logger?.logGnss(
                tSeconds, location.latitude, location.longitude,
                speed, bearing, accuracy, withheld = blackoutActive
            )

            // Always fed to the pipeline too. onGnss's own blackout branch decides
            // whether the fix reaches the filter or only the ground-truth
            // accounting - see FusionPipeline.onGnss's docs.
            pipeline?.onGnss(
                FusionPipeline.GnssFix(
                    tNs = location.elapsedRealtimeNanos,
                    latDeg = location.latitude,
                    lonDeg = location.longitude,
                    speedMps = speed,
                    bearingDeg = bearing,
                    accuracyM = accuracy
                )
            )

            publishStatus(pipeline?.describeFrontEnd() ?: "")
            publishTelemetry()
        }

        @Deprecated("Deprecated in API 29, still required on API 26 targets")
        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit
        override fun onProviderEnabled(provider: String) = Unit
        override fun onProviderDisabled(provider: String) = Unit
    }

    /**
     * Keeps [RecordingTelemetry] ticking at roughly 1 Hz for the whole session,
     * independent of what else is happening.
     *
     * Before this, a telemetry publish only happened as a side effect of a GNSS
     * fix arriving or a fused IMU cycle completing (`RUNNING` phase only - see
     * [maybeEmitImuSample]). That is fine once the pipeline is running, but it
     * means the Live screen goes completely silent - no elapsed time, no phase
     * change, nothing - for however long `LEVELING`/`WAITING_FOR_GNSS` takes
     * without a GNSS fix to drive an update. With GPS disabled at system level
     * (the exact case [RecordingTelemetry.gpsProviderEnabled] exists to catch)
     * that silence is permanent: no fix ever arrives, so nothing after the very
     * first publish at [startRecording] would ever have refreshed it. A driver
     * watching a frozen screen for the first 30+ seconds of a cold GNSS fix
     * cannot tell "working normally" from "stuck"; this heartbeat removes that
     * ambiguity by guaranteeing a fresh publish every second for as long as
     * [pipeline] is non-null, cheap since [publishTelemetry] is already cheap
     * (no sensor I/O, just a data class copy and a main-thread post).
     */
    private val heartbeat = object : Runnable {
        override fun run() {
            if (pipeline == null) return
            publishTelemetry()
            handler.postDelayed(this, 1000L)
        }
    }

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        thread = HandlerThread("sih26-sensor-ingest")
        thread.start()
        handler = Handler(thread.looper)
        sessionStore = SessionStore.open(this)

        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            // System-triggered restart after the process was killed
            // (START_STICKY delivers a null Intent here, distinct from any explicit
            // action). In-memory fusion/leveling state cannot be reconstructed, so
            // there is no session to resume - continuing to run would just be a
            // silent background service doing nothing with no way for the user to
            // stop it. Whatever was written to the JSONL before the kill survives
            // on disk regardless and is picked up as a recovered session (see
            // SessionStore.recoverOrphans) the next time the app is opened.
            if (pipeline == null) stopSelf()
            return START_NOT_STICKY
        }
        when (intent.action) {
            ACTION_START -> startRecording()
            ACTION_STOP -> stopRecording()
            ACTION_SET_BLACKOUT -> {
                val active = intent.getBooleanExtra(EXTRA_BLACKOUT_ACTIVE, false)
                // FusionPipeline is owned by the sensor handler thread. Move this
                // control event there too; otherwise a UI toggle can race an IMU
                // callback halfway through a filter step.
                handler.post {
                    val now = android.os.SystemClock.elapsedRealtimeNanos()
                    if (active && !blackoutActive) blackoutStartedNs = now
                    if (!active && blackoutActive) {
                        blackoutStartedNs?.let { blackoutDurationS += (now - it) / 1_000_000_000.0 }
                        blackoutStartedNs = null
                    }
                    blackoutActive = active
                    // Fusion timestamps use Android's elapsedRealtime clock. Java's
                    // nanoTime has no API guarantee of sharing that epoch,
                    // especially across suspend, so it must not be mixed into
                    // blackout accounting.
                    pipeline?.setBlackout(active, now)
                    publishTelemetry()
                }
            }
        }
        return START_STICKY
    }

    private fun startRecording() {
        // A repeat START intent must not create a second logger or re-register the
        // same listener. The UI normally prevents this; the service remains robust
        // to duplicate intents after process recreation.
        if (pipeline != null) return

        // startForeground() must be called promptly whenever the service was
        // reached via startForegroundService() (MainActivity always uses that),
        // regardless of what the validation below finds - the system enforces
        // this with a hard crash (ForegroundServiceDidNotStartInTimeException) if
        // it is skipped. The two validation failure branches below used to call
        // stopSelf() before ever reaching the startForeground() call further down
        // this function, which is exactly the sequence that crash requires: deny
        // the location permission (or start with GPS/sensors unavailable) and the
        // whole app would go down instead of showing the intended error message.
        // Calling it first with a "starting" notification, then tearing it down
        // cleanly on failure, satisfies the contract unconditionally.
        //
        // On API 34 (this app's targetSdk) calling startForeground() with a
        // declared foregroundServiceType="location" is itself gated on location
        // permission: with neither ACCESS_FINE_LOCATION nor ACCESS_COARSE_LOCATION
        // granted, the platform throws a SecurityException
        // (MissingForegroundServiceTypePermissionsException) out of this call,
        // before the explicit permission check below ever runs. MainActivity's
        // "Start recording" button is already gated on hasLocationPermission, so
        // this is not reachable from the normal UI flow, but the service also
        // accepts ACTION_START directly and permission can be revoked from system
        // Settings between the button press and the service actually starting -
        // both would have crashed the whole process here instead of reaching the
        // intended "location permission not granted" error path. Caught explicitly
        // so a missing permission always degrades to that message, never a crash.
        try {
            startForeground(NOTIFICATION_ID, buildNotification())
        } catch (e: SecurityException) {
            publishStatus("ERROR: location permission not granted (${e.message})")
            stopSelf()
            return
        }

        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            publishStatus("ERROR: location permission not granted")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }

        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        if (accelerometer == null || gyroscope == null) {
            publishStatus("ERROR: device is missing accelerometer or gyroscope")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "sih26:recording").apply {
            setReferenceCounted(false)
            // 6 hours: far beyond any real field test, just a backstop, not a
            // planned session length.
            acquire(6 * 60 * 60 * 1000L)
        }

        sessionStartElapsedNs = android.os.SystemClock.elapsedRealtimeNanos()
        lastAccel = null
        lastGyro = null
        latestMagnetometer = null
        blackoutActive = false
        blackoutStartedNs = null
        blackoutDurationS = 0.0
        finalBlackoutDriftM = null
        gnssFixes = 0
        distanceM = 0.0
        lastGnssTimestampNs = null
        lastGnssSpeedMps = null
        lastGnssAccuracyM = null
        lastSnapshot = null
        lastTelemetryPublishNs = 0L
        gnssSatellitesInView = null
        gnssSatellitesUsed = null
        fusionFailed = false
        // Stage 12: loads once per recording start; loadFromAssets never throws, so
        // a missing/corrupt model degrades to stage12Model = null (Channel P only)
        // rather than failing the whole recording start. See MotionSpeedNetOnnx's
        // class doc.
        //
        // Corridor: OverpassRoadNetworkProvider fetches once, lazily, on the
        // session's first GNSS fix (see CorridorChannel.onFirstFix) - constructing
        // it here just wires the (network-free at construction time) client in, it
        // does not touch the network yet.
        roadProvider?.close()
        val provider = OverpassRoadNetworkProvider()
        roadProvider = provider
        pipeline = FusionPipeline(
            stage12Model = MotionSpeedNetOnnx.loadFromAssets(this),
            roadNetworkProvider = provider
        )

        val (newSessionId, outFile) = sessionStore.newSessionFile()
        sessionId = newSessionId
        sessionStartedUtc = SessionStore.nowUtc()
        logger = SessionLogger(
            outFile,
            device = android.os.Build.MODEL ?: "unknown",
            notes = "recorded by SessionRecordingService"
        )

        sensorManager.registerListener(sensorListener, accelerometer, SAMPLING_PERIOD_US, handler)
        sensorManager.registerListener(sensorListener, gyroscope, SAMPLING_PERIOD_US, handler)
        if (magnetometer != null) {
            // Best effort, at whatever rate is convenient - nothing consumes this
            // stream, it is logged only so PS 26168's named inputs are all captured.
            sensorManager.registerListener(sensorListener, magnetometer, SensorManager.SENSOR_DELAY_NORMAL, handler)
        }

        locationManager.requestLocationUpdates(
            LocationManager.GPS_PROVIDER,
            GNSS_MIN_INTERVAL_MS,
            GNSS_MIN_DISTANCE_M,
            locationListener,
            thread.looper
        )
        // Best-effort: satellite visibility is diagnostic, not load-bearing, and
        // must never take a recording down if registration fails for any reason
        // (e.g. a chip/driver that does not support the callback).
        runCatching {
            if (Build.VERSION.SDK_INT >= 30) {
                // Run on the same sensor-ingest thread as everything else here
                // (see this class's doc point 4), not the main executor, so a
                // satellite-status update cannot race the IMU/GNSS state it reads
                // and writes alongside.
                locationManager.registerGnssStatusCallback({ command -> handler.post(command) }, gnssStatusCallback)
            } else {
                @Suppress("DEPRECATION")
                locationManager.registerGnssStatusCallback(gnssStatusCallback, handler)
            }
        }
        publishTelemetry()
        handler.removeCallbacks(heartbeat)
        handler.postDelayed(heartbeat, 1000L)
    }

    private fun stopRecording() {
        handler.removeCallbacks(heartbeat)
        sensorManager.unregisterListener(sensorListener)
        locationManager.removeUpdates(locationListener)
        runCatching { locationManager.unregisterGnssStatusCallback(gnssStatusCallback) }
        val droppedSamples = logger?.droppedSampleCount ?: 0
        val loggerFailed = logger?.failed ?: false
        logger?.close()
        val now = android.os.SystemClock.elapsedRealtimeNanos()
        if (blackoutActive) {
            blackoutStartedNs?.let { blackoutDurationS += (now - it) / 1_000_000_000.0 }
        }
        val id = sessionId
        val rawFile = logger?.outputFile
        if (id != null && rawFile != null) {
            sessionStore.save(SessionRecord(
                id = id, startedUtc = sessionStartedUtc,
                durationS = (now - sessionStartElapsedNs) / 1_000_000_000.0,
                imuSamples = pipeline?.imuSamples ?: 0, gnssFixes = gnssFixes,
                distanceM = distanceM, blackoutDurationS = blackoutDurationS,
                finalDriftM = finalBlackoutDriftM, rawFile = rawFile,
                droppedSamples = droppedSamples,
                loggerFailed = loggerFailed,
                fusionFailed = fusionFailed
            ))
        }
        logger = null
        pipeline = null
        roadProvider?.close()
        roadProvider = null
        blackoutActive = false
        releaseWakeLock()
        publishTelemetry()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    override fun onDestroy() {
        // Belt-and-suspenders: normally stopRecording() already did this, but
        // onDestroy can run without it (process death, a system-triggered kill
        // mid-recording) and an update delivered to a Handler on a thread that is
        // about to quit is harmless but pointless to leave registered.
        sensorManager.unregisterListener(sensorListener)
        locationManager.removeUpdates(locationListener)
        runCatching { locationManager.unregisterGnssStatusCallback(gnssStatusCallback) }
        logger?.close()
        roadProvider?.close()
        roadProvider = null
        releaseWakeLock()
        thread.quitSafely()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    /** Compose state may only change on its owning main thread. Sensor and GNSS
     * callbacks intentionally run on [thread], so cross the boundary exactly here. */
    private fun publishStatus(status: String) {
        Handler(Looper.getMainLooper()).post { statusListener?.invoke(status) }
    }

    private fun publishSnapshot(snapshot: FusionSnapshot?) {
        Handler(Looper.getMainLooper()).post { snapshotListener?.invoke(snapshot) }
    }

    private fun publishTelemetry() {
        val now = android.os.SystemClock.elapsedRealtimeNanos()
        val pipeline = pipeline
        val telemetry = RecordingTelemetry(
            recording = pipeline != null,
            phase = pipeline?.phase,
            waitingForMovement = pipeline?.waitingForMovingFix ?: false,
            blackout = blackoutActive,
            elapsedS = if (pipeline == null) 0.0 else (now - sessionStartElapsedNs) / 1_000_000_000.0,
            distanceM = distanceM, imuSamples = pipeline?.imuSamples ?: 0,
            gnssFixes = gnssFixes, gnssSpeedMps = lastGnssSpeedMps,
            gnssAccuracyM = lastGnssAccuracyM, snapshot = lastSnapshot,
            droppedSamples = logger?.droppedSampleCount ?: 0,
            loggerFailed = logger?.failed ?: false,
            fusionFailed = fusionFailed,
            gpsProviderEnabled = runCatching {
                locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
            }.getOrDefault(true),
            gnssSatellitesInView = gnssSatellitesInView,
            gnssSatellitesUsed = gnssSatellitesUsed,
            originLatDeg = pipeline?.localFrame?.lat0Deg,
            originLonDeg = pipeline?.localFrame?.lon0Deg
        )
        Handler(Looper.getMainLooper()).post { telemetryListener?.invoke(telemetry) }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW
        )
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        val pendingIntent = android.app.PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            android.app.PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(getString(R.string.notification_text))
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
