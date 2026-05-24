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
 * Đăng ký "chữ ký lắc" cá nhân.
 *
 * Hệ thống đo cách user lắc điện thoại trong 3 trial độc lập, lấy trung vị
 * số lần lắc làm baseline. Khi fallback, user phải lắc xấp xỉ đúng số lần đó
 * (±[ShakeDetector.SHAKE_TOLERANCE]). Đây là tín hiệu sinh trắc nhẹ — dựa
 * trên nhịp/biên độ lắc tự nhiên của user, không phải mật khẩu user nhớ.
 */
class FallbackEnrollActivity : AppCompatActivity() {
    private lateinit var tvTrialStatus: TextView
    private lateinit var tvCounter: TextView
    private lateinit var btnShake: Button
    private lateinit var progressBar: ProgressBar

    private lateinit var shakeDetector: ShakeDetector
    private lateinit var storage: PatternStorage

    private val trialCounts = mutableListOf<Int>()
    private var currentTrial = 0
    private var shaking = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_fallback_enroll)

        AuthenticationService.stop(this)

        tvTrialStatus = findViewById(R.id.feTrialStatus)
        tvCounter     = findViewById(R.id.feCounter)
        btnShake      = findViewById(R.id.feBtnShake)
        progressBar   = findViewById(R.id.feProgress)

        storage = PatternStorage(this)
        shakeDetector = ShakeDetector(this, onCountUpdated = { count ->
            runOnUiThread {
                tvCounter.text = count.toString()
                btnShake.text  = "Xong lắc  ($count lần)"
            }
        })

        progressBar.max = TOTAL_TRIALS
        progressBar.progress = 0

        btnShake.setOnClickListener {
            if (!shaking) startTrial() else finishTrial()
        }
    }

    override fun onDestroy() {
        shakeDetector.shutdown()
        super.onDestroy()
    }

    private fun startTrial() {
        shaking = true
        tvCounter.text = "0"
        tvTrialStatus.text = "Lần ${currentTrial + 1} / $TOTAL_TRIALS — lắc theo nhịp của bạn, rồi bấm 'Xong lắc'."
        btnShake.text = "Xong lắc  (0 lần)"

        shakeDetector.start(MAX_TRIAL_DURATION_MS)
    }

    private fun finishTrial() {
        if (!shaking) return
        shaking = false
        val count = shakeDetector.stop()
        trialCounts.add(count)
        currentTrial++
        progressBar.progress = currentTrial

        if (currentTrial >= TOTAL_TRIALS) {
            saveAndProceed()
        } else {
            btnShake.text = "Bắt đầu lắc"
            tvTrialStatus.text = "Lần $currentTrial xong (đếm: $count). Còn ${TOTAL_TRIALS - currentTrial} lần."
        }
    }

    private fun saveAndProceed() {
        val sorted = trialCounts.sorted()
        val median = sorted[sorted.size / 2]
        storage.savePattern(median)
        tvTrialStatus.text = "Đã lưu chữ ký lắc = $median lần (trung vị $TOTAL_TRIALS lần thử)."
        tvCounter.text = median.toString()
        btnShake.text = "Hoàn tất"
        btnShake.isEnabled = false
        Toast.makeText(this, "Đã đăng ký mẫu lắc ($median lần)", Toast.LENGTH_SHORT).show()
        btnShake.postDelayed({
            startActivity(Intent(this, QuizActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            })
        }, 1200L)
    }

    companion object {
        private const val TOTAL_TRIALS = 3
        private const val MAX_TRIAL_DURATION_MS = 5 * 60 * 1000L
    }
}
