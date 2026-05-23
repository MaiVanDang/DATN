package com.datn.authenticator.fallback

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.CountDownTimer
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import com.datn.authenticator.R
import com.datn.authenticator.service.AuthenticationService

/**
 * Full-screen overlay shown when the AuthenticationService transitions to UNKNOWN.
 *
 * UI flow (Mục 5.4):
 *   1) Show "Người dùng lạ — vui lòng xác thực bằng cử chỉ lắc"
 *   2) User taps "Bắt đầu xác thực" -> ShakeDetector starts (30 s window)
 *   3) Live counter shows current shake count
 *   4) On timeout OR user taps "Hoàn thành":
 *        - If verify() passes -> dismiss, show success toast, reset score state
 *        - Else, increment failed attempts. After 3 failures -> hand off to system PIN.
 */
class FallbackActivity : Activity() {

    private lateinit var titleText: TextView
    private lateinit var statusText: TextView
    private lateinit var counterText: TextView
    private lateinit var startButton: Button
    private lateinit var doneButton: Button
    private lateinit var cancelButton: Button

    private lateinit var shakeDetector: ShakeDetector
    private lateinit var storage: PatternStorage

    private var countdown: CountDownTimer? = null
    private var shakingActive = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Show on lock screen, keep screen on while user is shaking
        window.addFlags(
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                    WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
        )
        setContentView(R.layout.activity_fallback)

        titleText = findViewById(R.id.fallbackTitle)
        statusText = findViewById(R.id.fallbackStatus)
        counterText = findViewById(R.id.fallbackCounter)
        startButton = findViewById(R.id.btnStartShake)
        doneButton = findViewById(R.id.btnDoneShake)
        cancelButton = findViewById(R.id.btnCancel)

        storage = PatternStorage(this)
        shakeDetector = ShakeDetector(this, onCountUpdated = { count ->
            runOnUiThread { counterText.text = getString(R.string.fallback_counter, count) }
        })

        if (!storage.isEnrolled()) {
            statusText.text = getString(R.string.fallback_not_enrolled)
            startButton.isEnabled = false
            return
        }

        if (storage.failedAttempts() >= PatternStorage.MAX_FAILED_ATTEMPTS) {
            statusText.text = getString(R.string.fallback_max_failed)
            startButton.isEnabled = false
            // Per Mục 5.4.2 — hand off to system credentials. Demo: just show notice.
            return
        }

        startButton.setOnClickListener { startShakeCapture() }
        doneButton.setOnClickListener { finishShakeCapture(timedOut = false) }
        cancelButton.setOnClickListener {
            // User dismissed. We do NOT reset the score — service stays in UNKNOWN.
            finish()
        }
        doneButton.isEnabled = false
    }

    override fun onDestroy() {
        countdown?.cancel()
        shakeDetector.shutdown()
        super.onDestroy()
    }

    override fun onBackPressed() {
        // Block back button while shake is active so user can't accidentally bypass.
        if (!shakingActive) super.onBackPressed()
    }

    private fun startShakeCapture() {
        shakingActive = true
        startButton.isEnabled = false
        doneButton.isEnabled = true
        cancelButton.isEnabled = false
        statusText.text = getString(R.string.fallback_shake_now)
        counterText.text = getString(R.string.fallback_counter, 0)

        shakeDetector.start(maxDurationMs = ShakeDetector.DEFAULT_TIMEOUT_MS)

        countdown?.cancel()
        countdown = object : CountDownTimer(ShakeDetector.DEFAULT_TIMEOUT_MS, 1000L) {
            override fun onTick(remaining: Long) {
                titleText.text = getString(R.string.fallback_title_with_remaining, remaining / 1000)
            }

            override fun onFinish() {
                if (shakingActive) finishShakeCapture(timedOut = true)
            }
        }.start()
    }

    private fun finishShakeCapture(timedOut: Boolean) {
        if (!shakingActive) return
        shakingActive = false
        countdown?.cancel()
        val observedCount = shakeDetector.stop()
        val passed = storage.verify(observedCount)

        if (passed) {
            storage.resetFailedAttempts()
            // Reset aggregator → trạng thái TRUSTED ngay lập tức, không bị re-trigger
            AuthenticationService.instance?.onFallbackVerified()
            statusText.text = getString(R.string.fallback_pass, observedCount)
            setResult(Activity.RESULT_OK, Intent().putExtra(EXTRA_VERIFIED, true))
            finishWithDelay()
        } else {
            val failedCount = storage.recordFailedAttempt()
            statusText.text = if (timedOut) {
                getString(R.string.fallback_fail_timeout, observedCount, failedCount)
            } else {
                getString(R.string.fallback_fail_count, observedCount, failedCount)
            }
            counterText.text = getString(R.string.fallback_counter, observedCount)
            if (failedCount >= PatternStorage.MAX_FAILED_ATTEMPTS) {
                // Thông báo service chặn re-trigger — chờ PIN hệ thống
                AuthenticationService.instance?.onFallbackMaxFailed()
                statusText.append("\n" + getString(R.string.fallback_max_failed))
                setResult(Activity.RESULT_CANCELED)
                finishWithDelay()
            } else {
                // Allow retry
                startButton.isEnabled = true
                doneButton.isEnabled = false
                cancelButton.isEnabled = true
            }
        }
    }

    private fun finishWithDelay() {
        // Brief pause so the user sees the result message
        startButton.postDelayed({ if (!isFinishing) finish() }, 1500L)
    }

    companion object {
        const val EXTRA_VERIFIED = "verified"

        // Convenience to launch from anywhere
        fun launch(context: android.content.Context) {
            val intent = Intent(context, FallbackActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(intent)
        }
    }
}
