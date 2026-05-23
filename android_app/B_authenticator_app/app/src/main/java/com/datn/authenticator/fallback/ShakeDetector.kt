package com.datn.authenticator.fallback

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import kotlin.math.abs

/**
 * Peak-detection algorithm for the shake-gesture fallback (Mục 3.6 / 5.4.1).
 *
 * - Listens on the accelerometer X axis.
 * - A "shake" is counted when |acc_x| crosses [PEAK_THRESHOLD_MPS2] (≈ 0.8 g).
 * - Adjacent peaks within [DEBOUNCE_MS] of each other are merged.
 * - The detector runs until [stop] is called or the user-supplied [maxDurationMs]
 *   elapses, then returns the count.
 *
 * Counting rule (matches the report): registered_pattern is a digit 0–9, where 0
 * means "10 shakes" — but the conversion is the responsibility of caller code
 * (PatternStorage / FallbackActivity). This class returns raw shake count only.
 */
class ShakeDetector(
    private val context: Context,
    private val onCountUpdated: (Int) -> Unit = {},
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val handlerThread = HandlerThread("ShakeDetector").apply { start() }
    private val handler = Handler(handlerThread.looper)

    @Volatile private var counting = false
    @Volatile private var count = 0
    @Volatile private var lastPeakElapsedMs = 0L
    @Volatile private var lastSignSign = 0  // -1 / 0 / +1 — to require sign-flip between peaks

    /** Begin counting until [stop] or `maxDurationMs` elapses. */
    fun start(maxDurationMs: Long = 30_000L) {
        if (counting) return
        if (accelerometer == null) {
            Log.w(TAG, "No accelerometer — shake detection disabled")
            return
        }
        count = 0
        lastPeakElapsedMs = 0L
        lastSignSign = 0
        counting = true
        sensorManager.registerListener(
            this, accelerometer, SensorManager.SENSOR_DELAY_GAME, handler
        )
        handler.postDelayed({ stop() }, maxDurationMs)
    }

    fun stop(): Int {
        if (!counting) return count
        counting = false
        sensorManager.unregisterListener(this)
        return count
    }

    fun shutdown() {
        stop()
        handlerThread.quitSafely()
    }

    fun currentCount(): Int = count

    override fun onSensorChanged(event: SensorEvent) {
        if (!counting) return
        if (event.sensor.type != Sensor.TYPE_ACCELEROMETER) return

        val ax = event.values[0]
        val mag = abs(ax)
        if (mag < PEAK_THRESHOLD_MPS2) return

        val nowMs = SystemClock.elapsedRealtime()
        if (nowMs - lastPeakElapsedMs < DEBOUNCE_MS) return  // debounce

        val sign = if (ax > 0) 1 else -1
        // Optional refinement: require sign flip between counted peaks. This means
        // a sustained tilt does not count as multiple shakes. We keep it lenient
        // (allow same sign) for the demo to make UX easier, but the code path
        // is here if you want stricter counting.
        val requireSignFlip = false
        if (requireSignFlip && sign == lastSignSign) return

        lastPeakElapsedMs = nowMs
        lastSignSign = sign
        count += 1
        onCountUpdated(count)
        Log.d(TAG, "Peak detected — count=$count (|acc_x|=${"%.2f".format(mag)})")
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    companion object {
        private const val TAG = "ShakeDetector"
        const val PEAK_THRESHOLD_MPS2 = 8.0f      // ≈ 0.8 g per Bảng 5.4
        const val DEBOUNCE_MS = 200L              // Bảng 5.4
        const val DEFAULT_TIMEOUT_MS = 30_000L    // Bảng 5.4
        const val SHAKE_TOLERANCE = 1             // Bảng 5.4 — ±1 shake accepted
    }
}
