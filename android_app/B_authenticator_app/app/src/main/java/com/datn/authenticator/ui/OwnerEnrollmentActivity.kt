package com.datn.authenticator.ui

import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.datn.authenticator.R
import com.datn.authenticator.fallback.PatternStorage
import com.datn.authenticator.inference.FusionEngine
import com.datn.authenticator.service.AuthenticationService
import com.datn.authenticator.inference.InferenceEngine
import com.datn.authenticator.inference.NpyReader
import com.datn.authenticator.inference.OwnerProfile
import com.datn.authenticator.inference.RandomForestClassifier
import com.datn.authenticator.inference.SensorWindowCollector
import com.datn.authenticator.inference.TouchCollector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Biometric enrollment — two phases:
 *
 *   Phase 1 (IMU): collect 6 sensor windows → CNN embeddings → anchor list
 *   Phase 2 (touch): collect tap/scroll/key events via quiz UI
 *   Phase 3 (train): RF_inertial + RF_touch on-device using impostor pool assets
 *
 * Full profile (anchors + RF + fusion_w) saved to OwnerProfile.
 */
class OwnerEnrollmentActivity : AppCompatActivity() {

    // Phase 1 views
    private lateinit var btnStart: Button
    private lateinit var btnClear: Button
    private lateinit var btnDone: Button
    private lateinit var progress: ProgressBar
    private lateinit var status: TextView
    private lateinit var summary: TextView

    // Phase 2 views
    private lateinit var touchSection: LinearLayout
    private lateinit var touchStatus: TextView
    private lateinit var touchStats: TextView
    private lateinit var btnTrainRF: Button
    private lateinit var keyInput: EditText

    private lateinit var collector: SensorWindowCollector
    private lateinit var engine: InferenceEngine
    private lateinit var ownerProfile: OwnerProfile

    private var anchors = mutableListOf<FloatArray>()
    private var enrolling = false

    // ── Stats update timer ────────────────────────────────────────────────
    private val statsRunnable = object : Runnable {
        override fun run() {
            touchStats.text = "Tap: ${TouchCollector.tapCount()}  |  Cuộn: ${TouchCollector.scrollCount()}  |  Phím: ${TouchCollector.keyCount()}"
            touchStats.postDelayed(this, 500)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_owner_enrollment)

        // Phase 1
        btnStart = findViewById(R.id.btnStartEnroll)
        btnClear = findViewById(R.id.btnClearProfile)
        btnDone  = findViewById(R.id.btnDoneEnroll)
        progress = findViewById(R.id.enrollProgress)
        status   = findViewById(R.id.enrollStatusText)
        summary  = findViewById(R.id.enrollSummary)

        // Phase 2
        touchSection = findViewById(R.id.touchEnrollSection)
        touchStatus  = findViewById(R.id.touchStatusText)
        touchStats   = findViewById(R.id.touchStatsText)
        btnTrainRF   = findViewById(R.id.btnTrainRF)
        keyInput     = findViewById(R.id.touchKeyInput)

        // Auth service must not run during enrollment
        AuthenticationService.stop(this)

        collector    = SensorWindowCollector(this)
        engine       = InferenceEngine.load(this, useGpu = true)
        ownerProfile = OwnerProfile(this)

        progress.max = ANCHOR_COUNT
        progress.progress = 0

        btnStart.setOnClickListener { startEnrollment() }
        btnClear.setOnClickListener { clearProfile() }
        btnDone.setOnClickListener  { finish() }
        btnTrainRF.setOnClickListener { trainAndSave() }

        // Keystroke collection: track text insertions / deletions
        keyInput.addTextChangedListener(object : TextWatcher {
            private var prevLen = 0
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                prevLen = s?.length ?: 0
            }
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val newLen = s?.length ?: 0
                val isDelete = newLen < prevLen
                TouchCollector.onKeyInserted(isDelete)
                prevLen = newLen
            }
        })

        // If both biometric and shake pattern already enrolled → skip to quiz
        if (ownerProfile.getAnchors().isNotEmpty() && ownerProfile.getRfInertial()?.isTrained == true) {
            if (PatternStorage(this).isEnrolled()) {
                goToQuiz()
            } else {
                goToFallbackEnroll()
            }
            return
        }

        refreshSummary()
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        // Feed touch events to TouchCollector for Phase 2
        TouchCollector.onTouchEvent(ev)
        return super.dispatchTouchEvent(ev)
    }

    override fun onDestroy() {
        touchStats.removeCallbacks(statsRunnable)
        try { collector.shutdown() } catch (_: Exception) {}
        try { engine.close()       } catch (_: Exception) {}
        super.onDestroy()
    }

    // ── Phase 1: IMU enrollment ───────────────────────────────────────────

    private fun startEnrollment() {
        if (enrolling) return
        enrolling = true
        anchors.clear()
        TouchCollector.resetSession()
        btnStart.isEnabled = false
        btnClear.isEnabled = false
        touchSection.visibility = View.GONE
        progress.progress = 0
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

                status.text = "Bước 1 xong! Hãy tương tác với quiz bên dưới."
                toast("IMU enrollment xong. Hãy cuộn, chọn đáp án và gõ văn bản.")

                // Show Phase 2
                touchSection.visibility = View.VISIBLE
                TouchCollector.resetSession()
                touchStats.post(statsRunnable)

            } catch (e: Exception) {
                status.text = getString(R.string.enroll_owner_error, e.message ?: "?")
                resetUi()
            } finally {
                enrolling = false
                btnStart.isEnabled = true
                btnClear.isEnabled = true
            }
        }
    }

    // ── Phase 2 + 3: Touch collection + RF training ───────────────────────

    private fun trainAndSave() {
        if (anchors.isEmpty()) {
            toast("Hãy hoàn thành Bước 1 trước!")
            return
        }
        btnTrainRF.isEnabled = false
        touchStatus.text = "Đang huấn luyện mô hình Random Forest..."
        touchStats.removeCallbacks(statsRunnable)

        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.Default) { trainModels() }
                withContext(Dispatchers.Main) {
                    ownerProfile.save(anchors, result.rfInertial, result.rfTouch, result.fusionW)
                    toast("Enrollment hoàn tất! Đăng ký mẫu lắc tiếp theo...")
                    goToFallbackEnroll()
                }
            } catch (e: Exception) {
                touchStatus.text = "Lỗi: ${e.message}"
                toast("Lỗi training: ${e.message}")
            } finally {
                btnTrainRF.isEnabled = true
            }
        }
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

    private data class TrainResult(
        val rfInertial: RandomForestClassifier,
        val rfTouch: RandomForestClassifier?,
        val fusionW: Float,
    )

    private fun trainModels(): TrainResult {
        // ── Load impostor pool ──────────────────────────────────────────
        val poolInertialRaw = try {
            NpyReader.readFloat32_2D(this, "impostor_pool_inertial.npy")
        } catch (e: Exception) {
            android.util.Log.w(TAG, "impostor_pool_inertial.npy missing, using empty pool: ${e.message}")
            emptyArray()
        }
        val poolTouchRaw = try {
            NpyReader.readFloat32_2D(this, "impostor_pool_touch.npy")
        } catch (e: Exception) {
            android.util.Log.w(TAG, "impostor_pool_touch.npy missing: ${e.message}")
            emptyArray()
        }

        // ── RF_inertial ─────────────────────────────────────────────────
        val posEmbeds = anchors.toTypedArray()

        // Cap negative pool tại NEG_POOL_RATIO × số positive để RF không bị
        // áp đảo bởi negatives. Pool gốc từ training có thể chứa cả data của
        // chính owner → subsample nhỏ hơn giảm tác động của data leakage này.
        val maxNeg = posEmbeds.size * NEG_POOL_RATIO
        val negEmbeds: Array<FloatArray> = if (poolInertialRaw.size > maxNeg) {
            val rng = java.util.Random(42L)
            val idx = poolInertialRaw.indices.toMutableList().shuffled(rng).take(maxNeg)
            Array(maxNeg) { poolInertialRaw[idx[it]] }
        } else poolInertialRaw

        android.util.Log.i(TAG, "RF_inertial: ${posEmbeds.size} pos + ${negEmbeds.size} neg")

        val X_i = (posEmbeds + negEmbeds)
        val y_i = IntArray(X_i.size) { if (it < posEmbeds.size) 1 else 0 }

        val rfInertial = RandomForestClassifier(nEstimators = 200, minSamplesLeaf = 2)
        if (X_i.isNotEmpty() && X_i[0].isNotEmpty()) {
            rfInertial.fit(X_i, y_i)
        }

        // ── RF_touch ────────────────────────────────────────────────────
        var rfTouch: RandomForestClassifier? = null
        val touchVec = TouchCollector.buildFeatureVector()
        val touchScaler = InferenceEngine.loadTouchScaler(this)

        if (touchVec != null && touchScaler != null && poolTouchRaw.isNotEmpty()) {
            val (mean, scale) = touchScaler
            val scaledOwner = FloatArray(touchVec.size) { i ->
                val s = scale[i].takeIf { it > 0f } ?: 1f
                (touchVec[i] - mean[i]) / s
            }
            val X_t = arrayOf(scaledOwner) + poolTouchRaw
            val y_t = IntArray(X_t.size) { if (it == 0) 1 else 0 }

            val rf = RandomForestClassifier(nEstimators = 200, minSamplesLeaf = 1)
            rf.fit(X_t, y_t)
            rfTouch = rf
            android.util.Log.i(TAG, "RF_touch trained: 1 owner + ${poolTouchRaw.size} impostors")
        } else {
            android.util.Log.w(TAG, "Skipping RF_touch: touchVec=${touchVec != null}, scaler=${touchScaler != null}, pool=${poolTouchRaw.size}")
        }

        // ── Fusion weight: use 0.5 default (no held-out val on-device) ──
        val fusionW = FusionEngine.DEFAULT_W

        return TrainResult(rfInertial, rfTouch, fusionW)
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private fun clearProfile() {
        ownerProfile.clear()
        anchors.clear()
        progress.progress = 0
        touchSection.visibility = View.GONE
        touchStats.removeCallbacks(statsRunnable)
        TouchCollector.resetSession()
        refreshSummary()
        toast(getString(R.string.enroll_owner_cleared))
    }

    private fun resetUi() {
        enrolling = false
        btnStart.isEnabled = true
        btnClear.isEnabled = true
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

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    companion object {
        private const val TAG = "OwnerEnrollment"

        // RF cần đủ positive samples để học trong 128-D: 6 là quá ít.
        // 20 windows ≈ 40s đi bộ, cải thiện đáng kể khả năng phân biệt.
        private const val ANCHOR_COUNT = 20

        // Số negative samples tối đa = ANCHOR_COUNT × hệ số này.
        // Giữ nhỏ tránh RF bị áp đảo khi positive ít.
        private const val NEG_POOL_RATIO = 4
    }
}
