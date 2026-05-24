package com.datn.authenticator.ui

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.datn.authenticator.R
import com.datn.authenticator.inference.FusionEngine
import com.datn.authenticator.inference.InferenceEngine
import com.datn.authenticator.inference.NpyReader
import com.datn.authenticator.inference.OwnerProfile
import com.datn.authenticator.inference.RandomForestClassifier
import com.datn.authenticator.inference.TouchCollector
import com.datn.authenticator.service.AuthenticationService
import com.datn.authenticator.util.ContextMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TouchEnrollActivity : AppCompatActivity() {
    private enum class Task { TAP, SCROLL, KEYSTROKE }

    private lateinit var tvStepLabel: TextView
    private lateinit var pbStep: ProgressBar
    private lateinit var tvTaskProgress: TextView
    private lateinit var tvTaskUnit: TextView
    private lateinit var sectionTap: View
    private lateinit var btnTapTarget: Button
    private lateinit var sectionScroll: View
    private lateinit var sectionKeystroke: View
    private lateinit var etTyping: EditText
    private lateinit var btnNext: Button

    private lateinit var ownerProfile: OwnerProfile

    private var currentTask = Task.TAP
    private var tapBaseline = 0
    private var scrollBaseline = 0
    private var keyBaseline = 0

    private val handler = Handler(Looper.getMainLooper())
    private val progressRunnable = object : Runnable {
        override fun run() {
            updateProgress()
            handler.postDelayed(this, 200)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_touch_enroll)

        AuthenticationService.stop(this)

        tvStepLabel      = findViewById(R.id.teStepLabel)
        pbStep           = findViewById(R.id.teStepProgress)
        tvTaskProgress   = findViewById(R.id.teTaskProgress)
        tvTaskUnit       = findViewById(R.id.teTaskUnit)
        sectionTap       = findViewById(R.id.teSectionTap)
        btnTapTarget     = findViewById(R.id.teBtnTapTarget)
        sectionScroll    = findViewById(R.id.teSectionScroll)
        sectionKeystroke = findViewById(R.id.teSectionKeystroke)
        etTyping         = findViewById(R.id.teEtTyping)
        btnNext          = findViewById(R.id.teBtnNext)

        ownerProfile = OwnerProfile(this)

        etTyping.addTextChangedListener(object : TextWatcher {
            private var prevLen = 0
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                prevLen = s?.length ?: 0
            }
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val newLen = s?.length ?: 0
                TouchCollector.onKeyInserted(newLen < prevLen)
                prevLen = newLen
            }
        })

        btnNext.setOnClickListener { onNextClicked() }

        TouchCollector.resetSession()
        showTask(Task.TAP)
        handler.post(progressRunnable)
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        TouchCollector.onTouchEvent(ev)
        return super.dispatchTouchEvent(ev)
    }

    override fun onDestroy() {
        handler.removeCallbacks(progressRunnable)
        super.onDestroy()
    }

    private fun showTask(task: Task) {
        currentTask = task
        when (task) {
            Task.TAP -> {
                tapBaseline = TouchCollector.tapCount()
                sectionTap.visibility = View.VISIBLE
                sectionScroll.visibility = View.GONE
                sectionKeystroke.visibility = View.GONE
                tvStepLabel.text = "Bước 1 / 3: Nhấn màn hình"
                pbStep.progress = 1
                tvTaskUnit.text = "lần nhấn"
                btnNext.text = "Tiếp: Lướt màn hình  ›"
                btnNext.isEnabled = false
            }
            Task.SCROLL -> {
                scrollBaseline = TouchCollector.scrollCount()
                sectionTap.visibility = View.GONE
                sectionScroll.visibility = View.VISIBLE
                sectionKeystroke.visibility = View.GONE
                tvStepLabel.text = "Bước 2 / 3: Lướt màn hình"
                pbStep.progress = 2
                tvTaskUnit.text = "lần lướt"
                btnNext.text = "Tiếp: Gõ phím  ›"
                btnNext.isEnabled = false
            }
            Task.KEYSTROKE -> {
                keyBaseline = TouchCollector.keyCount()
                sectionTap.visibility = View.GONE
                sectionScroll.visibility = View.GONE
                sectionKeystroke.visibility = View.VISIBLE
                tvStepLabel.text = "Bước 3 / 3: Gõ phím"
                pbStep.progress = 3
                tvTaskUnit.text = "ký tự"
                btnNext.text = "Hoàn thành"
                btnNext.isEnabled = false
            }
        }
        updateProgress()
    }

    private fun updateProgress() {
        when (currentTask) {
            Task.TAP -> {
                val done = (TouchCollector.tapCount() - tapBaseline).coerceAtLeast(0)
                tvTaskProgress.text = "$done / $TAP_REQUIRED"
                btnNext.isEnabled = done >= TAP_REQUIRED
            }
            Task.SCROLL -> {
                val done = (TouchCollector.scrollCount() - scrollBaseline).coerceAtLeast(0)
                tvTaskProgress.text = "$done / $SCROLL_REQUIRED"
                btnNext.isEnabled = done >= SCROLL_REQUIRED
            }
            Task.KEYSTROKE -> {
                val done = (TouchCollector.keyCount() - keyBaseline).coerceAtLeast(0)
                tvTaskProgress.text = "$done / $KEY_REQUIRED"
                btnNext.isEnabled = done >= KEY_REQUIRED
            }
        }
    }

    private fun onNextClicked() {
        when (currentTask) {
            Task.TAP      -> showTask(Task.SCROLL)
            Task.SCROLL   -> showTask(Task.KEYSTROKE)
            Task.KEYSTROKE -> trainAndFinish()
        }
    }

    private fun trainAndFinish() {
        btnNext.isEnabled = false
        btnNext.text = "Đang huấn luyện…"
        val anchors = ownerProfile.getAnchors()
        if (anchors.isEmpty()) {
            toast("Lỗi: không tìm thấy dữ liệu IMU. Hãy quay lại đăng ký lại.")
            return
        }
        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.Default) { trainModels(anchors) }
                ownerProfile.save(anchors, result.rfInertial, result.rfTouch, result.fusionW)
                withContext(Dispatchers.Main) {
                    toast("Hoàn tất! Tiếp theo: đăng ký mẫu lắc.")
                    startActivity(Intent(this@TouchEnrollActivity, FallbackEnrollActivity::class.java).apply {
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                    })
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    toast("Lỗi training: ${e.message}")
                    btnNext.isEnabled = true
                    btnNext.text = "Hoàn thành"
                }
            }
        }
    }

    private data class TrainResult(
        val rfInertial: RandomForestClassifier,
        val rfTouch: RandomForestClassifier?,
        val fusionW: Float,
    )

    private fun trainModels(anchors: List<FloatArray>): TrainResult {
        val mode = ContextMode.loadOrDefault(this)
        val poolInertialPath = ContextMode.assetPath(mode, "impostor_pool_inertial.npy")
        val poolTouchPath    = ContextMode.assetPath(mode, "impostor_pool_touch.npy")

        val poolInertialRaw = try {
            NpyReader.readFloat32_2D(this, poolInertialPath)
        } catch (e: Exception) {
            Log.w(TAG, "$poolInertialPath missing: ${e.message}")
            emptyArray()
        }
        val poolTouchRaw = try {
            NpyReader.readFloat32_2D(this, poolTouchPath)
        } catch (e: Exception) {
            Log.w(TAG, "$poolTouchPath missing: ${e.message}")
            emptyArray()
        }

        val posEmbeds = anchors.toTypedArray()
        val maxNeg = posEmbeds.size * NEG_POOL_RATIO
        val negEmbeds: Array<FloatArray> = if (poolInertialRaw.size > maxNeg) {
            val rng = java.util.Random(42L)
            val idx = poolInertialRaw.indices.toMutableList().shuffled(rng).take(maxNeg)
            Array(maxNeg) { poolInertialRaw[idx[it]] }
        } else poolInertialRaw

        Log.i(TAG, "RF_inertial[${mode.key}]: ${posEmbeds.size} pos + ${negEmbeds.size} neg")
        val X_i = (posEmbeds + negEmbeds)
        val y_i = IntArray(X_i.size) { if (it < posEmbeds.size) 1 else 0 }
        val rfInertial = RandomForestClassifier(nEstimators = 200, minSamplesLeaf = 2)
        if (X_i.isNotEmpty() && X_i[0].isNotEmpty()) rfInertial.fit(X_i, y_i)

        var rfTouch: RandomForestClassifier? = null
        val touchVec = TouchCollector.buildFeatureVector()
        val touchScaler = InferenceEngine.loadTouchScaler(this, mode)

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
            Log.i(TAG, "RF_touch[${mode.key}] trained: 1 owner + ${poolTouchRaw.size} impostors")
        } else {
            Log.w(TAG, "Skip RF_touch: vec=${touchVec != null} scaler=${touchScaler != null} pool=${poolTouchRaw.size}")
        }

        return TrainResult(rfInertial, rfTouch, FusionEngine.DEFAULT_W)
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    companion object {
        private const val TAG = "TouchEnroll"
        const val TAP_REQUIRED    = 15
        const val SCROLL_REQUIRED = 8
        const val KEY_REQUIRED    = 60
        private const val NEG_POOL_RATIO = 4
    }
}
