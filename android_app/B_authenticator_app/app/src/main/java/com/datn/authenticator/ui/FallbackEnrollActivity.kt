package com.datn.authenticator.ui

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.datn.authenticator.R
import com.datn.authenticator.fallback.PatternStorage
import com.datn.authenticator.fallback.ShakeDetector
import com.datn.authenticator.service.AuthenticationService

/**
 * Đăng ký "mật khẩu lắc" dạng DÃY CHỮ SỐ (mỗi chữ số = số lần lắc trong 1 cụm,
 * dừng ~1s để sang chữ số kế). Nhập 2 lần phải KHỚP thì mới lưu — tránh sai
 * do nhịp lắc. Lưu băm SHA-256, không lưu dãy thô.
 */
class FallbackEnrollActivity : AppCompatActivity() {
    private lateinit var tvStatus: TextView
    private lateinit var tvCounter: TextView
    private lateinit var btnShake: Button
    private lateinit var progressBar: ProgressBar

    private lateinit var shakeDetector: ShakeDetector
    private lateinit var storage: PatternStorage

    private var firstEntry: List<Int>? = null
    private var shaking = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_fallback_enroll)
        AuthenticationService.stop(this)

        tvStatus    = findViewById(R.id.feTrialStatus)
        tvCounter   = findViewById(R.id.feCounter)
        btnShake    = findViewById(R.id.feBtnShake)
        progressBar = findViewById(R.id.feProgress)

        storage = PatternStorage(this)
        shakeDetector = ShakeDetector(
            this,
            onDigitsUpdated = { seq -> runOnUiThread { tvCounter.text = format(seq) } },
            onShake = { b -> runOnUiThread { tvCounter.text = "${prefix()}${b}…" } },
        )

        progressBar.max = 2
        progressBar.progress = 0
        tvStatus.text = "Đặt mật khẩu lắc: ${PatternStorage.MIN_DIGITS}–${PatternStorage.MAX_DIGITS} chữ số. " +
            "Mỗi chữ số là số lần lắc liên tiếp (1–9), dừng ~1s để sang chữ số kế."
        btnShake.text = "Bắt đầu lắc"
        btnShake.setOnClickListener { if (!shaking) startEntry() else finishEntry() }
    }

    override fun onDestroy() { shakeDetector.shutdown(); super.onDestroy() }

    private fun startEntry() {
        shaking = true
        tvCounter.text = "—"
        tvStatus.text = if (firstEntry == null) "Lần 1: lắc dãy của bạn rồi bấm 'Xong'."
                        else "Lần 2: lắc lại ĐÚNG dãy đó để xác nhận."
        btnShake.text = "Xong"
        shakeDetector.start(ShakeDetector.DEFAULT_TIMEOUT_MS)
    }

    private fun finishEntry() {
        if (!shaking) return
        shaking = false
        val seq = shakeDetector.stop()
        btnShake.text = "Bắt đầu lắc"

        try { PatternStorage.validate(seq) } catch (e: Exception) {
            tvStatus.text = "Dãy không hợp lệ: ${e.message}. Thử lại."
            firstEntry = null; progressBar.progress = 0
            return
        }

        if (firstEntry == null) {
            firstEntry = seq
            progressBar.progress = 1
            tvStatus.text = "Đã ghi lần 1: ${format(seq)}. Bấm để nhập lại xác nhận."
        } else if (firstEntry == seq) {
            storage.savePattern(seq)
            progressBar.progress = 2
            tvStatus.text = "Đã lưu mật khẩu lắc (${seq.size} chữ số)."
            tvCounter.text = format(seq)
            btnShake.isEnabled = false
            Toast.makeText(this, "Đăng ký thành công", Toast.LENGTH_SHORT).show()
            btnShake.postDelayed({
                startActivity(Intent(this, QuizActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                })
            }, 1200L)
        } else {
            tvStatus.text = "Hai lần không khớp (${format(firstEntry!!)} ≠ ${format(seq)}). Làm lại từ đầu."
            firstEntry = null; progressBar.progress = 0
        }
    }

    private fun prefix(): String {
        val s = shakeDetector.currentSequence()
        return if (s.isEmpty()) "" else s.joinToString(" - ") + " - "
    }
    private fun format(seq: List<Int>) = if (seq.isEmpty()) "—" else seq.joinToString(" - ")
}
