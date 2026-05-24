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

    fun start(maxDurationMs: Long = 30_000L) {
        if (counting) return
        if (accelerometer == null) {
            Log.w(TAG, "No accelerometer — shake detection disabled")
            return
        }
        count = 0
        lastPeakElapsedMs = 0L
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
        if (nowMs - lastPeakElapsedMs < DEBOUNCE_MS) return

        lastPeakElapsedMs = nowMs
        count += 1
        onCountUpdated(count)
        Log.d(TAG, "Peak detected — count=$count (|acc_x|=${"%.2f".format(mag)})")
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    companion object {
        private const val TAG = "ShakeDetector"
        const val PEAK_THRESHOLD_MPS2 = 8.0f
        const val DEBOUNCE_MS = 200L
        const val DEFAULT_TIMEOUT_MS = 30_000L
        const val SHAKE_TOLERANCE = 1
    }
}
