package com.datn.authenticator.inference

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import com.datn.authenticator.model.SensorWindow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

class SensorWindowCollector(context: Context) : SensorEventListener {
    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    private val handlerThread = HandlerThread("SensorWindowCollector").apply { start() }
    private val handler = Handler(handlerThread.looper)

    private val maxRawSamplesPerChannel = SAMPLE_RATE_HZ * WINDOW_SECONDS * 4
    private val accBuf = RawChannelBuffer(maxRawSamplesPerChannel)
    private val gyroBuf = RawChannelBuffer(maxRawSamplesPerChannel)
    private val magBuf = RawChannelBuffer(maxRawSamplesPerChannel)

    @Volatile private var capturing = false
    @Volatile private var captureStartElapsedMs: Long = 0L

    suspend fun collectOneWindow(): SensorWindow? = suspendCancellableCoroutine { cont ->
        if (accelerometer == null || gyroscope == null) {
            Log.w(TAG, "Required sensors unavailable (acc=${accelerometer != null}, gyro=${gyroscope != null})")
            cont.resume(null)
            return@suspendCancellableCoroutine
        }
        accBuf.clear(); gyroBuf.clear(); magBuf.clear()
        capturing = true
        captureStartElapsedMs = SystemClock.elapsedRealtime()

        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME, handler)
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME, handler)
        magnetometer?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME, handler) }

        val finishedRunnable = Runnable {
            if (!cont.isActive) return@Runnable
            capturing = false
            sensorManager.unregisterListener(this)
            val window = buildWindow(System.currentTimeMillis())
            cont.resume(window)
        }
        handler.postDelayed(finishedRunnable, WINDOW_SECONDS * 1000L)

        val timeoutRunnable = Runnable {
            if (!cont.isActive) return@Runnable
            if (capturing) {
                capturing = false
                sensorManager.unregisterListener(this)
                Log.w(TAG, "Timeout collecting sensor window")
                cont.resume(null)
            }
        }
        handler.postDelayed(timeoutRunnable, TIMEOUT_MS)

        cont.invokeOnCancellation {
            capturing = false
            sensorManager.unregisterListener(this)
            handler.removeCallbacks(finishedRunnable)
            handler.removeCallbacks(timeoutRunnable)
        }
    }

    fun shutdown() {
        sensorManager.unregisterListener(this)
        handlerThread.quitSafely()
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!capturing) return

        val tsMs = event.timestamp / 1_000_000L
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> accBuf.add(tsMs, event.values[0], event.values[1], event.values[2])
            Sensor.TYPE_GYROSCOPE -> gyroBuf.add(tsMs, event.values[0], event.values[1], event.values[2])
            Sensor.TYPE_MAGNETIC_FIELD -> magBuf.add(tsMs, event.values[0], event.values[1], event.values[2])
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun buildWindow(endTimestampMs: Long): SensorWindow? {
        val nAcc = accBuf.size()
        if (nAcc < SensorWindow.TIMESTEPS || gyroBuf.size() < 60) {
            Log.w(TAG, "Too few samples — acc=$nAcc, gyro=${gyroBuf.size()}. Discarding.")
            return null
        }

        // Lấy TRỰC TIẾP TIMESTEPS mẫu gia tốc kế thô gần nhất ở tần số gốc của
        // thiết bị. Gyro/mag được căn theo dấu thời gian của từng mẫu acc để
        // đồng bộ ba cảm biến bất đồng bộ về cùng hàng.
        val data = FloatArray(SensorWindow.TIMESTEPS * SensorWindow.CHANNELS)
        val start = nAcc - SensorWindow.TIMESTEPS
        for (i in 0 until SensorWindow.TIMESTEPS) {
            val idx = start + i
            val tMs = accBuf.timestampAt(idx)
            val a = accBuf.valueAt(idx)
            val row = i * SensorWindow.CHANNELS
            data[row + SensorWindow.CH_ACC_X] = a[0]
            data[row + SensorWindow.CH_ACC_Y] = a[1]
            data[row + SensorWindow.CH_ACC_Z] = a[2]
            gyroBuf.alignAt(tMs)?.let {
                data[row + SensorWindow.CH_GYRO_X] = it[0]
                data[row + SensorWindow.CH_GYRO_Y] = it[1]
                data[row + SensorWindow.CH_GYRO_Z] = it[2]
            }
            if (magBuf.size() > 0) magBuf.alignAt(tMs)?.let {
                data[row + SensorWindow.CH_MAG_X] = it[0]
                data[row + SensorWindow.CH_MAG_Y] = it[1]
                data[row + SensorWindow.CH_MAG_Z] = it[2]
            }
        }

        return SensorWindow(
            data = data,
            startTimestampMs = endTimestampMs - WINDOW_SECONDS * 1000L,
            endTimestampMs = endTimestampMs,
        )
    }

    private class RawChannelBuffer(capacity: Int) {
        private val ts = LongArray(capacity)
        private val x = FloatArray(capacity)
        private val y = FloatArray(capacity)
        private val z = FloatArray(capacity)
        private var head = 0
        private var n = 0

        fun clear() { head = 0; n = 0 }
        fun size() = n
        fun firstTimestamp(): Long = if (n > 0) ts[0] else 0L
        fun lastTimestamp(): Long = if (n > 0) ts[n - 1] else 0L
        fun timestampAt(i: Int): Long = ts[i]
        fun valueAt(i: Int): FloatArray = floatArrayOf(x[i], y[i], z[i])

        fun add(timestampMs: Long, vx: Float, vy: Float, vz: Float) {
            if (n >= ts.size) return
            ts[n] = timestampMs; x[n] = vx; y[n] = vy; z[n] = vz; n++
        }

        /**
         * Căn giá trị cảm biến về thời điểm targetMs (dấu thời gian của mẫu acc)
         * bằng nội suy tuyến tính giữa hai mẫu gần nhất, nhằm đồng bộ các cảm biến
         * bất đồng bộ về cùng một hàng. Đây là bước CĂN ĐỒNG BỘ, không phải chuẩn
         * hóa lại tần số. Trả null nếu buffer rỗng; ngoài biên thì kẹp về mẫu đầu/cuối.
         */
        fun alignAt(targetMs: Long): FloatArray? {
            if (n == 0) return null
            if (targetMs <= ts[0]) return floatArrayOf(x[0], y[0], z[0])
            if (targetMs >= ts[n - 1]) return floatArrayOf(x[n - 1], y[n - 1], z[n - 1])
            // tìm i sao cho ts[i] <= targetMs < ts[i+1] (buffer đã theo thứ tự thời gian)
            var lo = 0; var hi = n - 1
            while (lo + 1 < hi) {
                val mid = (lo + hi) ushr 1
                if (ts[mid] <= targetMs) lo = mid else hi = mid
            }
            val t0 = ts[lo]; val t1 = ts[lo + 1]
            val span = (t1 - t0).toFloat()
            val frac = if (span > 0f) (targetMs - t0).toFloat() / span else 0f
            return floatArrayOf(
                x[lo] + (x[lo + 1] - x[lo]) * frac,
                y[lo] + (y[lo + 1] - y[lo]) * frac,
                z[lo] + (z[lo + 1] - z[lo]) * frac,
            )
        }
    }

    companion object {
        private const val TAG = "SensorWindowCollector"
        const val SAMPLE_RATE_HZ = SensorWindow.SAMPLE_RATE_HZ
        const val WINDOW_SECONDS = 4
        const val TIMEOUT_MS = 9_000L
    }
}