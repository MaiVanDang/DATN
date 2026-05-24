package com.datn.authenticator.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorManager
import android.hardware.TriggerEvent
import android.hardware.TriggerEventListener
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.datn.authenticator.AuthenticatorApp
import com.datn.authenticator.fallback.FallbackActivity
import com.datn.authenticator.inference.InferenceEngine
import com.datn.authenticator.inference.ScoreAggregator
import com.datn.authenticator.inference.SensorWindowCollector
import com.datn.authenticator.inference.TouchCollector
import com.datn.authenticator.model.AuthState
import com.datn.authenticator.ui.QuizActivity
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

    private var warningTimerJob: Job? = null

    @Volatile private var fallbackGraceUntilMs: Long = 0L

    @Volatile private var fallbackBlocked: Boolean = false

    internal val _state = MutableStateFlow(AuthState.TRUSTED)
    val state: StateFlow<AuthState> = _state.asStateFlow()
    internal val _score = MutableStateFlow(0.5f)
    val score: StateFlow<Float> = _score.asStateFlow()

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "AuthenticationService onCreate")

        TouchCollector.resetSession()

        promoteToForeground(AuthState.TRUSTED, 0.5f)

        collector = SensorWindowCollector(this)
        inferenceEngine = InferenceEngine.load(this)
        touchScaler = InferenceEngine.loadTouchScaler(this)
        Log.i(TAG, "InferenceEngine ready: backend=${inferenceEngine?.backend}, mock=${inferenceEngine?.isMockMode}, touchScaler=${touchScaler != null}")

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        significantMotionSensor = sensorManager.getDefaultSensor(Sensor.TYPE_SIGNIFICANT_MOTION)
        significantMotionSensor?.let {
            sensorManager.requestTriggerSensor(significantMotionListener, it)
            Log.d(TAG, "Registered significant motion trigger")
        } ?: Log.w(TAG, "Significant motion sensor unavailable; relying on screen-on only")

        ContextCompat.registerReceiver(
            this, screenOnReceiver, IntentFilter(Intent.ACTION_SCREEN_ON), ContextCompat.RECEIVER_NOT_EXPORTED
        )

        captureLoopJob = scope.launch { runCaptureLoop() }

        requestCapture("startup")

        scope.launch {
            while (true) {
                delay(PERIODIC_CAPTURE_INTERVAL_MS)
                requestCapture("periodic")
            }
        }

        instance = this
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        promoteToForeground(_state.value, _score.value)
        return START_STICKY
    }

    override fun onTimeout(startId: Int) {
        Log.w(TAG, "Foreground service timeout reached (dataSync 6h cap) — stopping self")
        stopSelf()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        Log.w(TAG, "Foreground service timeout reached (type=$fgsType) — stopping self")
        stopSelf()
    }

    private fun promoteToForeground(state: AuthState, score: Float) {
        ServiceCompat.startForeground(
            this,
            NotificationHelper.NOTIFICATION_ID_SERVICE,
            buildNotification(state, score),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
        )
    }

    override fun onDestroy() {
        Log.i(TAG, "AuthenticationService onDestroy")
        try { unregisterReceiver(screenOnReceiver) } catch (_: Exception) {}
        try { significantMotionSensor?.let { sensorManager.cancelTriggerSensor(significantMotionListener, it) } } catch (_: Exception) {}
        warningTimerJob?.cancel()
        captureLoopJob?.cancel()
        scope.cancel()
        try { if (::collector.isInitialized) collector.shutdown() } catch (_: Exception) {}
        try { inferenceEngine?.close() } catch (_: Exception) {}
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

    private suspend fun runCaptureLoop() {
        while (true) {
            val reason = pendingCaptureReason
            if (reason == null) {
                delay(POLL_INTERVAL_MS)
                continue
            }
            pendingCaptureReason = null
            lastCaptureElapsedMs = SystemClock.elapsedRealtime()

            val window = withTimeoutOrNull(SensorWindowCollector.TIMEOUT_MS + 1000L) {
                collector.collectOneWindow()
            }
            if (window == null) {
                Log.w(TAG, "Capture failed/timeout (reason=$reason)")
                continue
            }

            val engine = inferenceEngine ?: continue

            val touchVec = TouchCollector.buildFeatureVector()

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

            if (newState == AuthState.TRUSTED) {
                inferenceEngine?.adaptiveBuffer()?.maybeAdd(
                    embed = result.embedding,
                    fusedScore = result.probabilityLegitimate,
                )
            }

            when (newState) {
                AuthState.TRUSTED -> {
                    warningTimerJob?.cancel()
                    warningTimerJob = null
                }
                AuthState.WARNING -> {
                    if (warningTimerJob == null || !warningTimerJob!!.isActive) {
                        Log.i(TAG, "WARNING — bắt đầu đếm ngược ${WARNING_TIMEOUT_MS / 1000}s")
                        warningTimerJob = scope.launch {
                            delay(WARNING_TIMEOUT_MS)
                            if (_state.value != AuthState.TRUSTED) {
                                Log.i(TAG, "WARNING timeout — launching fallback")
                                inferenceEngine?.adaptiveBuffer()?.onFallbackTriggered()
                                launchFallbackActivity()
                            }
                            warningTimerJob = null
                        }
                    }
                }
                AuthState.UNKNOWN -> {
                    warningTimerJob?.cancel()
                    warningTimerJob = null
                    if (previousState != AuthState.UNKNOWN) {
                        Log.i(TAG, "UNKNOWN — launching fallback immediately")
                        inferenceEngine?.adaptiveBuffer()?.onFallbackTriggered()
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

    fun onFallbackVerified() {
        fallbackBlocked = false
        fallbackGraceUntilMs = SystemClock.elapsedRealtime() + FALLBACK_GRACE_MS
        warningTimerJob?.cancel(); warningTimerJob = null
        aggregator.reset()

        val highScore = aggregator.push(1.0f)
        _score.value = highScore
        _state.value = aggregator.currentState()
        updateNotification(_state.value, highScore)
        Log.i(TAG, "onFallbackVerified — score reset, state=${_state.value}")
    }

    fun onFallbackMaxFailed() {
        fallbackBlocked = true
        Log.i(TAG, "onFallbackMaxFailed — fallback blocked until PIN unlocks")
    }

    private fun formatPct(v: Float) = "${(v * 100).toInt()}%"

    companion object {
        private const val TAG = "${AuthenticatorApp.TAG}/Service"

        private const val COOLDOWN_MS = 5_000L
        private const val POLL_INTERVAL_MS = 250L
        private const val PERIODIC_CAPTURE_INTERVAL_MS = 4_000L
        private const val WARNING_TIMEOUT_MS = 15_000L
        private const val FALLBACK_GRACE_MS = 30_000L

        @Volatile
        var instance: AuthenticationService? = null
            internal set

        fun isRunning(): Boolean = instance != null

        fun currentScore(): Float? = instance?._score?.value
        fun currentState(): AuthState? = instance?._state?.value

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
