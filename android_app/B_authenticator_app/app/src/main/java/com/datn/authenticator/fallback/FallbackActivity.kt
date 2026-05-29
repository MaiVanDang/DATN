package com.datn.authenticator.fallback

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.CountDownTimer
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import com.datn.authenticator.R
import com.datn.authenticator.service.AuthenticationService

/**
 * Màn hình fallback khi nghi ngờ người lạ: yêu cầu nhập lại DÃY LẮC bí mật.
 * Tái sử dụng layout activity_fallback (các view id giữ nguyên); ô counter
 * giờ hiển thị dãy chữ số đang nhập, ví dụ "3 - 1 - 4".
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
        shakeDetector = ShakeDetector(
            this,
            onDigitsUpdated = { seq -> runOnUiThread { counterText.text = format(seq) } },
            onShake = { b -> runOnUiThread { counterText.text = "${currentPrefix()}${b}…" } },
        )

        if (!storage.isEnrolled()) {
            statusText.text = "Chưa đăng ký mẫu lắc."
            startButton.isEnabled = false
            return
        }
        if (storage.failedAttempts() >= PatternStorage.MAX_FAILED_ATTEMPTS) {
            statusText.text = "Đã khoá: vượt số lần thử cho phép."
            startButton.isEnabled = false
            return
        }

        statusText.text = "Nhập lại dãy lắc bí mật (${storage.registeredLength()} chữ số)."
        startButton.setOnClickListener { startCapture() }
        doneButton.setOnClickListener { finishCapture(timedOut = false) }
        cancelButton.setOnClickListener { finish() }
        doneButton.isEnabled = false
    }

    override fun onDestroy() {
        countdown?.cancel(); shakeDetector.shutdown(); super.onDestroy()
    }

    override fun onBackPressed() { if (!shakingActive) super.onBackPressed() }

    private fun startCapture() {
        shakingActive = true
        startButton.isEnabled = false
        doneButton.isEnabled = true
        cancelButton.isEnabled = false
        statusText.text = "Lắc theo dãy của bạn: mỗi chữ số là số lần lắc, dừng ~1s để sang chữ số kế."
        counterText.text = "—"
        shakeDetector.start(ShakeDetector.DEFAULT_TIMEOUT_MS)

        countdown?.cancel()
        countdown = object : CountDownTimer(ShakeDetector.DEFAULT_TIMEOUT_MS, 1000L) {
            override fun onTick(remaining: Long) { titleText.text = "Còn ${remaining / 1000}s" }
            override fun onFinish() { if (shakingActive) finishCapture(timedOut = true) }
        }.start()
    }

    private fun finishCapture(timedOut: Boolean) {
        if (!shakingActive) return
        shakingActive = false
        countdown?.cancel()
        val observed = shakeDetector.stop()
        counterText.text = format(observed)

        if (storage.verify(observed)) {
            storage.resetFailedAttempts()
            AuthenticationService.instance?.onFallbackVerified()
            statusText.text = "Xác thực thành công."
            setResult(Activity.RESULT_OK, Intent().putExtra(EXTRA_VERIFIED, true))
            finishWithDelay()
        } else {
            val failed = storage.recordFailedAttempt()
            statusText.text = (if (timedOut) "Hết giờ. " else "Sai dãy. ") +
                "Nhập: ${format(observed)} (lần thử $failed/${PatternStorage.MAX_FAILED_ATTEMPTS})"
            if (failed >= PatternStorage.MAX_FAILED_ATTEMPTS) {
                AuthenticationService.instance?.onFallbackMaxFailed()
                statusText.append("\nĐã khoá — yêu cầu PIN hệ thống.")
                setResult(Activity.RESULT_CANCELED)
                finishWithDelay()
            } else {
                startButton.isEnabled = true
                doneButton.isEnabled = false
                cancelButton.isEnabled = true
            }
        }
    }

    private fun currentPrefix(): String {
        val s = shakeDetector.currentSequence()
        return if (s.isEmpty()) "" else s.joinToString(" - ") + " - "
    }

    private fun format(seq: List<Int>): String =
        if (seq.isEmpty()) "—" else seq.joinToString(" - ")

    private fun finishWithDelay() =
        startButton.postDelayed({ if (!isFinishing) finish() }, 1500L)

    companion object {
        const val EXTRA_VERIFIED = "verified"
        fun launch(context: Context) {
            context.startActivity(Intent(context, FallbackActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        }
    }
}
