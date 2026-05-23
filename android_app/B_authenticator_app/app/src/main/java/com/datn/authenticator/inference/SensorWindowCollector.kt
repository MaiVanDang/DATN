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

/**
 * Captures one 200-sample (4-second @ 50 Hz) window of inertial data and
 * returns it as a SensorWindow ready to feed into the InferenceEngine.
 *
 * Implements the per-session sensor capture described in Mục 5.3.2 of the
 * thesis report. Sample rate is set to SENSOR_DELAY_GAME (target ~50 Hz)
 * but Android does not guarantee a hard deadline — we resample-by-binning
 * on the fly so the output is always exactly 200 samples × 9 channels even
 * if the OS delivered slightly more or fewer raw events.
 *
 * Note on resampling here vs. preprocessing-time resample (Mục 4.4.4):
 *   - The training pipeline does linear interpolation onto a strict 50 Hz grid.
 *   - At inference time we use simpler nearest-neighbor binning to keep the
 *     critical path fast (<1 ms). The accuracy difference on a single 2 s
 *     window is negligible per pilot tests, and the wall-clock cost of
 *     numpy.interp-style code in pure Kotlin would be 10× larger.
 */
class SensorWindowCollector(context: Context) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    private val handlerThread = HandlerThread("SensorWindowCollector").apply { start() }
    private val handler = Handler(handlerThread.looper)

    // Per-channel ring buffers — we sample at SENSOR_DELAY_GAME (~50Hz)
    // but Android may deliver up to ~200Hz on some devices; we cap.
    private val maxRawSamplesPerChannel = SAMPLE_RATE_HZ * WINDOW_SECONDS * 4
    private val accBuf = RawChannelBuffer(maxRawSamplesPerChannel)
    private val gyroBuf = RawChannelBuffer(maxRawSamplesPerChannel)
    private val magBuf = RawChannelBuffer(maxRawSamplesPerChannel)

    @Volatile private var capturing = false
    @Volatile private var captureStartElapsedMs: Long = 0L

    /**
     * Suspends until exactly [WINDOW_SECONDS] of data have been gathered,
     * or until [TIMEOUT_MS] elapses. Returns null on timeout / missing sensor.
     */
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

    /** Stop background thread; call from Service.onDestroy(). */
    fun shutdown() {
        sensorManager.unregisterListener(this)
        handlerThread.quitSafely()
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!capturing) return
        // Android sensor events use SystemClock.elapsedRealtimeNanos timestamps
        val tsMs = event.timestamp / 1_000_000L  // nanos -> ms (relative, but consistent across sensors)
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> accBuf.add(tsMs, event.values[0], event.values[1], event.values[2])
            Sensor.TYPE_GYROSCOPE -> gyroBuf.add(tsMs, event.values[0], event.values[1], event.values[2])
            Sensor.TYPE_MAGNETIC_FIELD -> magBuf.add(tsMs, event.values[0], event.values[1], event.values[2])
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /**
     * Build a single (200, 9) window by binning raw samples into 200 equal-time bins
     * over the 4-second capture period. If a bin is empty (sensor dropped a sample)
     * we forward-fill from the previous bin; if the very first bin is empty we
     * back-fill from the next non-empty bin.
     */
    private fun buildWindow(endTimestampMs: Long): SensorWindow? {
        if (accBuf.size() < 60 || gyroBuf.size() < 60) {
            Log.w(TAG, "Too few samples — acc=${accBuf.size()}, gyro=${gyroBuf.size()}. Discarding.")
            return null
        }

        val data = FloatArray(SensorWindow.TIMESTEPS * SensorWindow.CHANNELS)
        // Range used for binning — span from first acc timestamp to last
        val tStart = accBuf.firstTimestamp()
        val tEnd = accBuf.lastTimestamp()
        if (tEnd <= tStart) return null

        val binWidth = (tEnd - tStart).toFloat() / SensorWindow.TIMESTEPS

        fillChannel(data, accBuf, tStart, binWidth, SensorWindow.CH_ACC_X, SensorWindow.CH_ACC_Y, SensorWindow.CH_ACC_Z)
        fillChannel(data, gyroBuf, tStart, binWidth, SensorWindow.CH_GYRO_X, SensorWindow.CH_GYRO_Y, SensorWindow.CH_GYRO_Z)
        if (magBuf.size() > 0) {
            fillChannel(data, magBuf, tStart, binWidth, SensorWindow.CH_MAG_X, SensorWindow.CH_MAG_Y, SensorWindow.CH_MAG_Z)
        }
        // mag channels stay 0.0f if magnetometer unavailable — matches training fallback (Mục 3.3.3)

        return SensorWindow(
            data = data,
            startTimestampMs = endTimestampMs - WINDOW_SECONDS * 1000L,
            endTimestampMs = endTimestampMs,
        )
    }

    private fun fillChannel(
        out: FloatArray,
        buf: RawChannelBuffer,
        tStart: Long,
        binWidth: Float,
        chX: Int, chY: Int, chZ: Int,
    ) {
        var lastX = 0f; var lastY = 0f; var lastZ = 0f
        var anyFilled = false

        for (binIdx in 0 until SensorWindow.TIMESTEPS) {
            val binStart = tStart + (binIdx * binWidth).toLong()
            val binEnd = tStart + ((binIdx + 1) * binWidth).toLong()
            // Average all samples in this bin; if empty, use last-known values
            val avg = buf.averageInRange(binStart, binEnd)
            if (avg != null) {
                lastX = avg[0]; lastY = avg[1]; lastZ = avg[2]; anyFilled = true
            }
            val rowOffset = binIdx * SensorWindow.CHANNELS
            out[rowOffset + chX] = lastX
            out[rowOffset + chY] = lastY
            out[rowOffset + chZ] = lastZ
        }

        // If the FIRST few bins were empty (sensor not yet warmed up), back-fill.
        // We do a second pass forward; entries written in the loop already have lastX/Y/Z set
        // properly once anyFilled=true, so only the leading zeros need patching.
        if (anyFilled) {
            // Find first non-zero (or first-filled) index
            var firstFilledBin = 0
            while (firstFilledBin < SensorWindow.TIMESTEPS) {
                val rowOffset = firstFilledBin * SensorWindow.CHANNELS
                if (out[rowOffset + chX] != 0f || out[rowOffset + chY] != 0f || out[rowOffset + chZ] != 0f) break
                firstFilledBin++
            }
            if (firstFilledBin in 1 until SensorWindow.TIMESTEPS) {
                val srcOffset = firstFilledBin * SensorWindow.CHANNELS
                val srcX = out[srcOffset + chX]
                val srcY = out[srcOffset + chY]
                val srcZ = out[srcOffset + chZ]
                for (i in 0 until firstFilledBin) {
                    val rowOffset = i * SensorWindow.CHANNELS
                    out[rowOffset + chX] = srcX
                    out[rowOffset + chY] = srcY
                    out[rowOffset + chZ] = srcZ
                }
            }
        }
    }

    /**
     * Append-only ring buffer of (timestamp_ms, x, y, z) tuples for one sensor.
     * Backed by parallel primitive arrays for GC-friendliness.
     */
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

        fun add(timestampMs: Long, vx: Float, vy: Float, vz: Float) {
            if (n >= ts.size) return  // drop overflow
            ts[n] = timestampMs; x[n] = vx; y[n] = vy; z[n] = vz; n++
        }

        /** Mean of (x,y,z) for samples whose timestamp ∈ [start, end). null if empty. */
        fun averageInRange(start: Long, end: Long): FloatArray? {
            var sx = 0f; var sy = 0f; var sz = 0f; var k = 0
            for (i in 0 until n) {
                val t = ts[i]
                if (t in start until end) {
                    sx += x[i]; sy += y[i]; sz += z[i]; k++
                }
            }
            return if (k == 0) null else floatArrayOf(sx / k, sy / k, sz / k)
        }
    }

    companion object {
        private const val TAG = "SensorWindowCollector"
        const val SAMPLE_RATE_HZ = SensorWindow.SAMPLE_RATE_HZ
        const val WINDOW_SECONDS = 4
        const val TIMEOUT_MS = 9_000L
    }
}
