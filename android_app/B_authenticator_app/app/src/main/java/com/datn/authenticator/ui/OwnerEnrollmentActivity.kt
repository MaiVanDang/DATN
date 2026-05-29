package com.datn.authenticator.ui

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.datn.authenticator.R
import com.datn.authenticator.fallback.PatternStorage
import com.datn.authenticator.inference.InferenceEngine
import com.datn.authenticator.inference.NpyReader
import com.datn.authenticator.inference.OwnerProfile
import com.datn.authenticator.inference.RandomForestClassifier
import com.datn.authenticator.inference.SensorWindowCollector
import com.datn.authenticator.service.AuthenticationService
import com.datn.authenticator.util.ContextMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class OwnerEnrollmentActivity : AppCompatActivity() {
    private lateinit var btnStart: Button
    private lateinit var btnChangeMode: Button
    private lateinit var progress: ProgressBar
    private lateinit var status: TextView
    private lateinit var summary: TextView
    private lateinit var modeIndicator: TextView

    private lateinit var collector: SensorWindowCollector
    private lateinit var engine: InferenceEngine
    private lateinit var ownerProfile: OwnerProfile

    private var anchors = mutableListOf<FloatArray>()
    private var enrolling = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextMode.loadSaved(this) == null) {
            startActivity(Intent(this, ModeSelectActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            })
            finish()
            return
        }

        ownerProfile = OwnerProfile(this)
        val storedAnchors = ownerProfile.getAnchors()
        if (storedAnchors.isNotEmpty()) {
            when {
                ownerProfile.getRfInertial()?.isTrained == true && PatternStorage(this).isEnrolled() -> {
                    goToQuiz(); return
                }
                ownerProfile.getRfInertial()?.isTrained == true -> {
                    goToFallbackEnroll(); return
                }
            }
        }

        setContentView(R.layout.activity_owner_enrollment)

        btnStart       = findViewById(R.id.btnStartEnroll)
        btnChangeMode  = findViewById(R.id.btnChangeMode)
        progress       = findViewById(R.id.enrollProgress)
        status         = findViewById(R.id.enrollStatusText)
        summary        = findViewById(R.id.enrollSummary)
        modeIndicator  = findViewById(R.id.modeIndicator)

        AuthenticationService.stop(this)

        collector    = SensorWindowCollector(this)
        engine       = InferenceEngine.load(this)

        progress.max = ANCHOR_COUNT
        progress.progress = 0

        btnStart.setOnClickListener { startEnrollment() }
        btnChangeMode.setOnClickListener { confirmChangeMode() }

        refreshModeIndicator()
        refreshSummary()
    }

    override fun onDestroy() {
        try { if (::collector.isInitialized) collector.shutdown() } catch (_: Exception) {}
        try { if (::engine.isInitialized) engine.close()       } catch (_: Exception) {}
        super.onDestroy()
    }

    private fun startEnrollment() {
        if (enrolling) return
        enrolling = true
        anchors.clear()
        progress.progress = 0
        btnStart.isEnabled = false
        btnChangeMode.isEnabled = false
        status.text = getString(R.string.enroll_owner_starting)

        lifecycleScope.launch {
            try {
                for (i in 1..ANCHOR_COUNT) {
                    status.text = getString(R.string.enroll_owner_capturing, i, ANCHOR_COUNT)
                    val window = withContext(Dispatchers.Default) { collector.collectOneWindow() }
                    if (window == null) {
                        status.text = getString(R.string.enroll_owner_failed_sensor)
                        toast(getString(R.string.enroll_owner_failed_sensor))
                        resetUi()
                        return@launch
                    }
                    val embed = withContext(Dispatchers.Default) { engine.extractEmbedding(window) }
                    anchors.add(embed)
                    progress.progress = i
                }

                status.text = "Đã thu xong ${anchors.size} mẫu IMU. Đang lưu..."
                val rfInertial = withContext(Dispatchers.Default) { trainRfInertial(anchors) }
                withContext(Dispatchers.Default) {
                    ownerProfile.save(anchors, rfInertial, null, 1f)
                }
                toast("Enrollment xong! Tiếp theo: đăng ký mẫu lắc.")
                goToFallbackEnroll()
            } catch (e: Exception) {
                status.text = getString(R.string.enroll_owner_error, e.message ?: "?")
                resetUi()
            } finally {
                enrolling = false
                btnStart.isEnabled = true
                btnChangeMode.isEnabled = true
            }
        }
    }

    private fun trainRfInertial(anchors: List<FloatArray>): RandomForestClassifier {
        val mode = ContextMode.loadOrDefault(this)
        val poolPath = ContextMode.assetPath(mode, "impostor_pool_inertial.npy")
        val poolRaw = try {
            NpyReader.readFloat32_2D(this, poolPath)
        } catch (e: Exception) {
            emptyArray()
        }
        val pos = anchors.toTypedArray()
        val maxNeg = pos.size * NEG_POOL_RATIO
        val neg: Array<FloatArray> = if (poolRaw.size > maxNeg) {
            val rng = java.util.Random(42L)
            val idx = poolRaw.indices.toMutableList().shuffled(rng).take(maxNeg)
            Array(maxNeg) { poolRaw[idx[it]] }
        } else poolRaw
        val x = (pos + neg)
        val y = IntArray(x.size) { if (it < pos.size) 1 else 0 }
        val rf = RandomForestClassifier(nEstimators = 200, minSamplesLeaf = 2)
        if (x.isNotEmpty() && x[0].isNotEmpty()) rf.fit(x, y)
        return rf
    }

    private fun goToFallbackEnroll() {
        startActivity(Intent(this, FallbackEnrollActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        })
    }

    private fun goToQuiz() {
        startActivity(Intent(this, QuizActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        })
    }

    private fun resetUi() {
        enrolling = false
        btnStart.isEnabled = true
        btnChangeMode.isEnabled = true
        refreshSummary()
    }

    private fun refreshSummary() {
        val n = ownerProfile.getAnchors().size
        val hasRf = ownerProfile.getRfInertial()?.isTrained == true
        summary.text = if (n > 0) {
            val rfLabel = if (hasRf) "+ RF" else "(anchors only)"
            "Đã enroll: $n anchor $rfLabel"
        } else {
            getString(R.string.enroll_owner_summary_empty)
        }
    }

    private fun refreshModeIndicator() {
        val mode = ContextMode.loadOrDefault(this)
        val modeLabel = when (mode) {
            ContextMode.WALKING -> getString(R.string.mode_walking_title)
            ContextMode.ALL     -> getString(R.string.mode_all_title)
        }
        modeIndicator.text = getString(R.string.enroll_owner_mode_indicator, modeLabel)
    }

    private fun confirmChangeMode() {
        AlertDialog.Builder(this)
            .setTitle(R.string.mode_change_dialog_title)
            .setMessage(R.string.mode_change_dialog_msg)
            .setPositiveButton(R.string.mode_change_dialog_confirm) { _, _ ->
                ownerProfile.clear()
                PatternStorage(this).clear()
                ContextMode.clear(this)
                AuthenticationService.stop(this)
                startActivity(Intent(this, ModeSelectActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                })
                finish()
            }
            .setNegativeButton(R.string.btn_cancel, null)
            .show()
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    companion object {
        private const val ANCHOR_COUNT = 20
        private const val NEG_POOL_RATIO = 4
    }
}
