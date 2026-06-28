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
 * Phát hiện cử chỉ lắc và tách thành dãy chữ số: mỗi đỉnh gia tốc vượt ngưỡng là một
 * lần lắc; một cụm lắc liên tiếp khi dừng quá [DIGIT_GAP_MS] được chốt thành một chữ
 * số (số lần lắc, kẹp 1..9). [onDigitsUpdated]/[onShake] cập nhật UI theo thời gian thực.
 */
class ShakeDetector(
    private val context: Context,
    private val onDigitsUpdated: (List<Int>) -> Unit = {},
    private val onShake: (currentBurst: Int) -> Unit = {},
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val handlerThread = HandlerThread("ShakeDetector").apply { start() }
    private val handler = Handler(handlerThread.looper)

    @Volatile private var counting = false
    @Volatile private var burstCount = 0
    @Volatile private var lastPeakMs = 0L
    private val sequence = mutableListOf<Int>()

    private val commitDigit = Runnable { commitCurrentBurst() }

    fun start(maxDurationMs: Long = DEFAULT_TIMEOUT_MS) {
        if (counting) return
        if (accelerometer == null) { Log.w(TAG, "No accelerometer"); return }
        sequence.clear(); burstCount = 0; lastPeakMs = 0L; counting = true
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME, handler)
        handler.postDelayed({ stop() }, maxDurationMs)
    }

    /** Chốt cụm cuối (nếu có) và trả về dãy chữ số đã ghi. */
    fun stop(): List<Int> {
        if (!counting) return sequence.toList()
        counting = false
        handler.removeCallbacks(commitDigit)
        commitCurrentBurst()
        sensorManager.unregisterListener(this)
        return sequence.toList()
    }

    fun shutdown() { stop(); handlerThread.quitSafely() }

    fun currentSequence(): List<Int> = sequence.toList()

    private fun commitCurrentBurst() {
        if (burstCount > 0) {
            sequence.add(burstCount.coerceIn(1, 9))
            burstCount = 0
            onDigitsUpdated(sequence.toList())
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!counting || event.sensor.type != Sensor.TYPE_ACCELEROMETER) return
        val mag = abs(
            Math.sqrt((event.values[0] * event.values[0] +
                    event.values[1] * event.values[1] +
                    event.values[2] * event.values[2]).toDouble()).toFloat() - 9.81f
        )
        if (mag < PEAK_THRESHOLD_MPS2) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastPeakMs < DEBOUNCE_MS) return
        lastPeakMs = now
        burstCount += 1
        onShake(burstCount)
        handler.removeCallbacks(commitDigit)
        handler.postDelayed(commitDigit, DIGIT_GAP_MS)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    companion object {
        private const val TAG = "ShakeDetector"
        const val PEAK_THRESHOLD_MPS2 = 6.0f
        const val DEBOUNCE_MS = 200L
        const val DIGIT_GAP_MS = 1200L
        const val DEFAULT_TIMEOUT_MS = 60_000L
    }
}
