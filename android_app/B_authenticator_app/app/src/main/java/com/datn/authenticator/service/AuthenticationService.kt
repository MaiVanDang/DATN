package com.datn.authenticator.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorManager
import android.hardware.TriggerEvent
import android.hardware.TriggerEventListener
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.core.content.ContextCompat
import com.datn.authenticator.AuthenticatorApp
import com.datn.authenticator.fallback.FallbackActivity
import com.datn.authenticator.inference.InferenceEngine
import com.datn.authenticator.ui.QuizActivity
import com.datn.authenticator.inference.ScoreAggregator
import com.datn.authenticator.inference.SensorWindowCollector
import com.datn.authenticator.inference.TouchCollector
import com.datn.authenticator.model.AuthState
import com.datn.authenticator.util.NotificationHelper
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

/**
 * AuthenticationService is the running heart of BioAuth Authenticator (per Mục 5.3.1).
 *
 * It owns the lifecycle of:
 *   - the SensorWindowCollector (event-triggered, see Mục 5.3.2)
 *   - the InferenceEngine (TFLite, Mục 5.5)
 *   - the ScoreAggregator (5-window EMA, Mục 5.3.3)
 *
 * Triggers a sensor capture session when EITHER:
 *   - the screen turns on (Intent.ACTION_SCREEN_ON), or
 *   - the OS reports significant motion (TriggerEvent on TYPE_SIGNIFICANT_MOTION).
 *
 * Cooldown: 5 seconds between consecutive captures (Mục 5.3.2) so we don't
 * burn battery while the user is actively walking.
 *
 * Exposes [state] as a Kotlin StateFlow so QuizActivity can observe and re-render.
 * The Service also fires FallbackActivity directly when the state transitions to UNKNOWN.
 */
class AuthenticationService : Service() {

    private val serviceJob = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.Default + serviceJob)
    private var captureLoopJob: Job? = null

    private lateinit var collector: SensorWindowCollector
    private var inferenceEngine: InferenceEngine? = null
    private val aggregator = ScoreAggregator()
    private var touchScaler: Pair<FloatArray, FloatArray>? = null

    private lateinit var sensorManager: SensorManager
    private var significantMotionSensor: Sensor? = null
    private val significantMotionListener = object : TriggerEventListener() {
        override fun onTrigger(event: TriggerEvent?) {
            Log.d(TAG, "Significant motion -> request capture")
            requestCapture("significant_motion")
            // Trigger sensors must be re-registered after each fire
            significantMotionSensor?.let { sensorManager.requestTriggerSensor(this, it) }
        }
    }

    private val screenOnReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == Intent.ACTION_SCREEN_ON) {
                Log.d(TAG, "Screen on -> request capture")
                requestCapture("screen_on")
            }
        }
    }

    @Volatile private var lastCaptureElapsedMs: Long = 0L
    @Volatile private var pendingCaptureReason: String? = null

    // WARNING countdown: nếu không hồi phục về TRUSTED trong 15s → fallback
    private var warningTimerJob: Job? = null

    // Grace period sau khi FallbackActivity vừa được launch — tránh spam popup
    @Volatile private var fallbackGraceUntilMs: Long = 0L

    // Blocked sau 3 lần fail shake — chờ user mở PIN hệ thống
    @Volatile private var fallbackBlocked: Boolean = false

    internal val _state = MutableStateFlow(AuthState.TRUSTED)
    val state: StateFlow<AuthState> = _state.asStateFlow()
    internal val _score = MutableStateFlow(0.5f)
    val score: StateFlow<Float> = _score.asStateFlow()

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "AuthenticationService onCreate")

        // 1) Promote to foreground IMMEDIATELY (Android 14 requires <5 s)
        startForeground(NotificationHelper.NOTIFICATION_ID_SERVICE, buildNotification(AuthState.TRUSTED, 0.5f))

        // 2) Wire up sensor / inference subsystem
        collector = SensorWindowCollector(this)
        inferenceEngine = InferenceEngine.load(this, useGpu = true)
        touchScaler = InferenceEngine.loadTouchScaler(this)
        Log.i(TAG, "InferenceEngine ready: backend=${inferenceEngine?.backend}, mock=${inferenceEngine?.isMockMode}, touchScaler=${touchScaler != null}")

        // 3) Hook event triggers
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        significantMotionSensor = sensorManager.getDefaultSensor(Sensor.TYPE_SIGNIFICANT_MOTION)
        significantMotionSensor?.let {
            sensorManager.requestTriggerSensor(significantMotionListener, it)
            Log.d(TAG, "Registered significant motion trigger")
        } ?: Log.w(TAG, "Significant motion sensor unavailable; relying on screen-on only")

        ContextCompat.registerReceiver(
            this, screenOnReceiver, IntentFilter(Intent.ACTION_SCREEN_ON), ContextCompat.RECEIVER_NOT_EXPORTED
        )

        // 4) Start the capture loop
        captureLoopJob = scope.launch { runCaptureLoop() }

        // 5) Bootstrap: do one immediate capture so the demo doesn't sit idle
        requestCapture("startup")

        // 6) Periodic capture for demo — every 4s so UI score updates regularly
        scope.launch {
            while (true) {
                delay(4_000L)
                requestCapture("periodic-demo")
            }
        }

        instance = this
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Service is sticky — the OS may restart us if memory is reclaimed
        return START_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "AuthenticationService onDestroy")
        try { unregisterReceiver(screenOnReceiver) } catch (_: Exception) {}
        try { significantMotionSensor?.let { sensorManager.cancelTriggerSensor(significantMotionListener, it) } } catch (_: Exception) {}
        warningTimerJob?.cancel()
        captureLoopJob?.cancel()
        scope.cancel()
        collector.shutdown()
        inferenceEngine?.close()
        instance = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun requestCapture(reason: String) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastCaptureElapsedMs < COOLDOWN_MS) {
            Log.d(TAG, "Capture cooldown ($reason) — ${(COOLDOWN_MS - (now - lastCaptureElapsedMs))} ms remaining")
            return
        }
        pendingCaptureReason = reason
    }

    /**
     * The capture loop polls a `pendingCaptureReason` flag every 250 ms.
     * When set, it kicks off one window capture + inference, updates score,
     * and may launch FallbackActivity. The polling approach keeps the loop
     * simple and lets ScreenOn/SignificantMotion be lock-free producers.
     */
    private suspend fun runCaptureLoop() {
        while (true) {
            val reason = pendingCaptureReason
            if (reason == null) {
                delay(POLL_INTERVAL_MS)
                continue
            }
            pendingCaptureReason = null
            lastCaptureElapsedMs = SystemClock.elapsedRealtime()

            // Capture one 2 s window
            val window = withTimeoutOrNull(SensorWindowCollector.TIMEOUT_MS + 1000L) {
                collector.collectOneWindow()
            }
            if (window == null) {
                Log.w(TAG, "Capture failed/timeout (reason=$reason)")
                continue
            }

            val engine = inferenceEngine ?: continue

            val touchVec = TouchCollector.buildFeatureVector()

            // Use RF + touch fusion if RF is trained; otherwise cosine fallback
            val fused = engine.predictFused(window, touchVec, touchScaler)
            val result = fused.inertial
            val scoreToAggregate = fused.fusedScore
            val newScore = aggregator.push(scoreToAggregate)
            val newState = aggregator.currentState()

            Log.i(TAG, "[$reason] p_inertial=${"%.3f".format(result.probabilityLegitimate)} " +
                    "p_touch=${fused.pTouch?.let { "%.3f".format(it) } ?: "n/a"} " +
                    "fused=${"%.3f".format(fused.fusedScore)} w=${"%.2f".format(fused.fusionW)} " +
                    "S=${"%.3f".format(newScore)} state=$newState " +
                    "lat=${"%.1f".format(result.totalLatencyMs)}ms")

            _score.value = newScore
            val previousState = _state.value
            _state.value = newState
            updateNotification(newState, newScore)

            when (newState) {
                AuthState.TRUSTED -> {
                    // Hồi phục → huỷ countdown WARNING nếu đang chạy
                    warningTimerJob?.cancel()
                    warningTimerJob = null
                }
                AuthState.WARNING -> {
                    // Chỉ bắt đầu đếm ngược nếu chưa có timer đang chạy
                    if (warningTimerJob == null || !warningTimerJob!!.isActive) {
                        Log.i(TAG, "WARNING — bắt đầu đếm ngược ${WARNING_TIMEOUT_MS / 1000}s")
                        warningTimerJob = scope.launch {
                            delay(WARNING_TIMEOUT_MS)
                            if (_state.value != AuthState.TRUSTED) {
                                Log.i(TAG, "WARNING timeout — launching fallback")
                                launchFallbackActivity()
                            }
                            warningTimerJob = null
                        }
                    }
                }
                AuthState.UNKNOWN -> {
                    // UNKNOWN: huỷ timer WARNING (tránh double-launch) + kích hoạt ngay
                    warningTimerJob?.cancel()
                    warningTimerJob = null
                    if (previousState != AuthState.UNKNOWN) {
                        Log.i(TAG, "UNKNOWN — launching fallback immediately")
                        launchFallbackActivity()
                    }
                }
            }
        }
    }

    private fun updateNotification(state: AuthState, score: Float) {
        val nm = androidx.core.app.NotificationManagerCompat.from(this)
        if (nm.areNotificationsEnabled()) {
            nm.notify(NotificationHelper.NOTIFICATION_ID_SERVICE, buildNotification(state, score))
        }
    }

    private fun buildNotification(state: AuthState, score: Float): Notification {
        val tapIntent = Intent(this, QuizActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pi = PendingIntent.getActivity(
            this, 0, tapIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val (title, text, iconRes) = when (state) {
            AuthState.TRUSTED -> Triple("BioAuth — Trusted", "Confidence ${formatPct(score)}", android.R.drawable.ic_lock_idle_lock)
            AuthState.WARNING -> Triple("BioAuth — Warning", "Confidence ${formatPct(score)} — monitoring", android.R.drawable.ic_dialog_alert)
            AuthState.UNKNOWN -> Triple("BioAuth — Unknown user", "Tap to authenticate", android.R.drawable.ic_dialog_alert)
        }

        return androidx.core.app.NotificationCompat.Builder(this, NotificationHelper.CHANNEL_SERVICE)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(iconRes)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .setContentIntent(pi)
            .setPriority(androidx.core.app.NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun launchFallbackActivity() {
        if (fallbackBlocked) {
            Log.d(TAG, "Fallback blocked (max failures reached — waiting for PIN)")
            return
        }
        val now = SystemClock.elapsedRealtime()
        if (now < fallbackGraceUntilMs) {
            Log.d(TAG, "Fallback in grace period — ${fallbackGraceUntilMs - now}ms remaining")
            return
        }
        fallbackGraceUntilMs = now + FALLBACK_GRACE_MS
        val intent = Intent(this, FallbackActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        try {
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Could not launch FallbackActivity", e)
        }
    }

    /** Gọi từ FallbackActivity khi shake xác thực thành công. */
    fun onFallbackVerified() {
        fallbackBlocked = false
        fallbackGraceUntilMs = SystemClock.elapsedRealtime() + FALLBACK_GRACE_MS
        warningTimerJob?.cancel(); warningTimerJob = null
        aggregator.reset()
        // Push score cao để state chuyển ngay về TRUSTED
        val highScore = aggregator.push(1.0f)
        _score.value = highScore
        _state.value = aggregator.currentState()
        updateNotification(_state.value, highScore)
        Log.i(TAG, "onFallbackVerified — score reset, state=${_state.value}")
    }

    /** Gọi từ FallbackActivity khi đã hết 3 lần thử — chờ PIN hệ thống. */
    fun onFallbackMaxFailed() {
        fallbackBlocked = true
        Log.i(TAG, "onFallbackMaxFailed — fallback blocked until PIN unlocks")
    }

    private fun formatPct(v: Float) = "${(v * 100).toInt()}%"

    companion object {
        private const val TAG = "${AuthenticatorApp.TAG}/Service"

        // Demo: capture every COOLDOWN_MS instead of waiting on motion events,
        // so the score visibly evolves during a 60-second hội đồng demo.
        // In production this would be 5_000L per Mục 5.3.2.
        private const val COOLDOWN_MS = 5_000L
        private const val POLL_INTERVAL_MS = 250L
        private const val WARNING_TIMEOUT_MS = 15_000L
        private const val FALLBACK_GRACE_MS = 30_000L  // tránh re-trigger 30s sau khi launch

        @Volatile
        var instance: AuthenticationService? = null
            internal set

        /** Whether the service is currently running in this process. */
        fun isRunning(): Boolean = instance != null

        fun currentScore(): Float? = instance?._score?.value
        fun currentState(): AuthState? = instance?._state?.value

        /** Convenience accessor for UI to observe state without binding. */
        fun observeState(): StateFlow<AuthState>? = instance?.state
        fun observeScore(): StateFlow<Float>? = instance?.score

        fun start(context: Context) {
            val intent = Intent(context, AuthenticationService::class.java)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, AuthenticationService::class.java))
        }
    }
}
