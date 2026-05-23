package com.datn.authenticator.ui

import android.content.Intent
import android.os.Bundle
import android.os.CountDownTimer
import android.widget.Button
import android.widget.NumberPicker
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.datn.authenticator.R
import com.datn.authenticator.fallback.PatternStorage
import com.datn.authenticator.fallback.ShakeDetector

/**
 * Shake-pattern enrollment — runs between biometric enrollment and the quiz.
 * User picks a secret digit (1–9), shakes 3 trials → median saved to PatternStorage.
 * After saving, navigates to QuizActivity (where AuthenticationService starts).
 */
class FallbackEnrollActivity : AppCompatActivity() {

    private lateinit var numberPicker: NumberPicker
    private lateinit var tvTrialStatus: TextView
    private lateinit var tvCounter: TextView
    private lateinit var btnShake: Button
    private lateinit var progressBar: ProgressBar

    private lateinit var shakeDetector: ShakeDetector
    private lateinit var storage: PatternStorage

    private val trialCounts = mutableListOf<Int>()
    private var currentTrial = 0
    private var shaking = false
    private var countdown: CountDownTimer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_fallback_enroll)

        numberPicker   = findViewById(R.id.feNumberPicker)
        tvTrialStatus  = findViewById(R.id.feTrialStatus)
        tvCounter      = findViewById(R.id.feCounter)
        btnShake       = findViewById(R.id.feBtnShake)
        progressBar    = findViewById(R.id.feProgress)

        storage = PatternStorage(this)
        shakeDetector = ShakeDetector(this, onCountUpdated = { count ->
            runOnUiThread { tvCounter.text = count.toString() }
        })

        numberPicker.minValue = 1
        numberPicker.maxValue = 9
        numberPicker.value = 3
        numberPicker.wrapSelectorWheel = false

        progressBar.max = TOTAL_TRIALS
        progressBar.progress = 0

        btnShake.setOnClickListener { startTrial() }
    }

    override fun onDestroy() {
        countdown?.cancel()
        shakeDetector.shutdown()
        super.onDestroy()
    }

    private fun startTrial() {
        if (shaking) return
        shaking = true
        btnShake.isEnabled = false
        numberPicker.isEnabled = false
        tvCounter.text = "0"
        tvTrialStatus.text = "Lần ${currentTrial + 1} / $TOTAL_TRIALS — lắc đi!"

        shakeDetector.start(TRIAL_DURATION_MS)

        countdown = object : CountDownTimer(TRIAL_DURATION_MS, 1000L) {
            override fun onTick(remaining: Long) {
                btnShake.text = "${remaining / 1000}s"
            }
            override fun onFinish() { finishTrial() }
        }.start()
    }

    private fun finishTrial() {
        if (!shaking) return
        shaking = false
        countdown?.cancel()
        val count = shakeDetector.stop()
        trialCounts.add(count)
        currentTrial++
        progressBar.progress = currentTrial

        if (currentTrial >= TOTAL_TRIALS) {
            saveAndProceed()
        } else {
            btnShake.text = "Bắt đầu lắc"
            btnShake.isEnabled = true
            tvTrialStatus.text = "Lần $currentTrial xong (đếm: $count). Còn ${TOTAL_TRIALS - currentTrial} lần."
        }
    }

    private fun saveAndProceed() {
        val sorted = trialCounts.sorted()
        val median = sorted[sorted.size / 2]
        val digit = numberPicker.value
        storage.savePattern(digit, median)
        tvTrialStatus.text = "Đã lưu! Số bí mật = $digit, mẫu lắc = $median lần."
        tvCounter.text = median.toString()
        Toast.makeText(this, "Đã đăng ký mẫu lắc ($median lần)", Toast.LENGTH_SHORT).show()
        btnShake.postDelayed({
            startActivity(Intent(this, QuizActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            })
        }, 1200L)
    }

    companion object {
        private const val TOTAL_TRIALS = 3
        private const val TRIAL_DURATION_MS = 10_000L
    }
}
