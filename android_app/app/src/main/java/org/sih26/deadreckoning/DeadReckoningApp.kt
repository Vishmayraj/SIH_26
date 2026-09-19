package org.sih26.deadreckoning

import android.app.Application
import org.osmdroid.config.Configuration

/** Exists as a named class (rather than the default android.app.Application) because
 * the manifest already references it and swapping in shared state later - a crash
 * reporter, a settings store - should not require touching the manifest again.
 *
 * The one real thing it does today: point osmdroid's tile cache at this app's own
 * files directory before any [org.sih26.deadreckoning.ui.TrajectoryCanvas] gets a
 * chance to construct a MapView. Without this, osmdroid falls back to a
 * world-readable path on shared storage that is not guaranteed to exist on modern
 * Android (WRITE_EXTERNAL_STORAGE is not requested), and tiles downloaded once would
 * not be cached for the next launch. */
class DeadReckoningApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Configuration.getInstance().load(this, getSharedPreferences("osmdroid", MODE_PRIVATE))
        Configuration.getInstance().osmdroidBasePath = getDir("osmdroid", MODE_PRIVATE)
        Configuration.getInstance().osmdroidTileCache = getDir("osmdroid", MODE_PRIVATE)
        Configuration.getInstance().userAgentValue = packageName
    }
}
