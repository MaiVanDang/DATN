package com.datn.authenticator.ui

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.datn.authenticator.R
import com.datn.authenticator.fallback.PatternStorage
import com.datn.authenticator.inference.OwnerProfile
import com.datn.authenticator.inference.TouchCollector
import com.datn.authenticator.model.AuthState
import com.datn.authenticator.service.AuthenticationService

/**
 * Quiz screen shown after enrollment. AuthenticationService runs throughout,
 * monitoring the user's inertial + touch behaviour while they interact with quiz.
 *
 * Score / state banner at the top updates every second from the running service.
 * FallbackActivity is launched by the service itself on UNKNOWN/WARNING-timeout.
 */
class QuizActivity : AppCompatActivity() {

    private lateinit var tvState: TextView
    private lateinit var tvScore: TextView
    private lateinit var tvQuestion: TextView
    private lateinit var tvQuestionNum: TextView
    private lateinit var rgAnswers: RadioGroup
    private lateinit var rb1: RadioButton
    private lateinit var rb2: RadioButton
    private lateinit var rb3: RadioButton
    private lateinit var rb4: RadioButton
    private lateinit var btnNext: Button
    private lateinit var tvResult: TextView
    private lateinit var etNotes: EditText
    private lateinit var tvKeyStats: TextView
    private lateinit var btnReenroll: Button

    private var currentQuestion = 0
    private var answered = false

    private val uiHandler = Handler(Looper.getMainLooper())

    private val statusRunnable = object : Runnable {
        override fun run() {
            updateAuthStatus()
            uiHandler.postDelayed(this, 1000L)
        }
    }

    private val keyStatsRunnable = object : Runnable {
        override fun run() {
            tvKeyStats.text = "Tap: ${TouchCollector.tapCount()}  |  Cuộn: ${TouchCollector.scrollCount()}  |  Phím: ${TouchCollector.keyCount()}"
            uiHandler.postDelayed(this, 500L)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_quiz)

        tvState       = findViewById(R.id.quizTvState)
        tvScore       = findViewById(R.id.quizTvScore)
        tvQuestion    = findViewById(R.id.quizTvQuestion)
        tvQuestionNum = findViewById(R.id.quizTvQuestionNum)
        rgAnswers     = findViewById(R.id.quizRgAnswers)
        rb1           = findViewById(R.id.quizRb1)
        rb2           = findViewById(R.id.quizRb2)
        rb3           = findViewById(R.id.quizRb3)
        rb4           = findViewById(R.id.quizRb4)
        btnNext       = findViewById(R.id.quizBtnNext)
        tvResult      = findViewById(R.id.quizTvResult)
        etNotes       = findViewById(R.id.quizEtNotes)
        tvKeyStats    = findViewById(R.id.quizTvKeyStats)
        btnReenroll   = findViewById(R.id.quizBtnReenroll)

        etNotes.addTextChangedListener(object : TextWatcher {
            private var prevLen = 0
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) { prevLen = s?.length ?: 0 }
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val newLen = s?.length ?: 0
                TouchCollector.onKeyInserted(newLen < prevLen)
                prevLen = newLen
            }
        })

        btnNext.setOnClickListener { handleNextButton() }

        btnReenroll.setOnClickListener {
            AuthenticationService.stop(this)
            OwnerProfile(this).clear()
            PatternStorage(this).clear()
            startActivity(Intent(this, OwnerEnrollmentActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            })
        }

        loadQuestion(0)
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        TouchCollector.onTouchEvent(ev)
        return super.dispatchTouchEvent(ev)
    }

    override fun onResume() {
        super.onResume()
        AuthenticationService.start(this)
        uiHandler.post(statusRunnable)
        uiHandler.post(keyStatsRunnable)
    }

    override fun onPause() {
        super.onPause()
        uiHandler.removeCallbacks(statusRunnable)
        uiHandler.removeCallbacks(keyStatsRunnable)
    }

    private fun updateAuthStatus() {
        val state = AuthenticationService.currentState() ?: return
        val scoreVal = AuthenticationService.currentScore() ?: return

        tvState.text = state.name
        val stateColor = when (state) {
            AuthState.TRUSTED -> ContextCompat.getColor(this, R.color.state_trusted)
            AuthState.WARNING -> ContextCompat.getColor(this, R.color.state_warning)
            AuthState.UNKNOWN -> ContextCompat.getColor(this, R.color.state_unknown)
        }
        tvState.setTextColor(stateColor)

        tvScore.text = "Tin cậy: ${"%.0f".format(scoreVal * 100)}%"
        tvScore.setTextColor(stateColor)

        // Update status bar background to reflect urgency
        val bar = findViewById<View>(R.id.quizStatusBar)
        bar.setBackgroundColor(when (state) {
            AuthState.TRUSTED -> 0xFFE8F5E9.toInt()
            AuthState.WARNING -> 0xFFFFF3E0.toInt()
            AuthState.UNKNOWN -> 0xFFFFEBEE.toInt()
        })
    }

    private fun loadQuestion(idx: Int) {
        val q = QUESTIONS[idx % QUESTIONS.size]
        tvQuestionNum.text = "Câu ${(idx % QUESTIONS.size) + 1} / ${QUESTIONS.size}"
        tvQuestion.text = q.text
        rb1.text = q.options[0]
        rb2.text = q.options[1]
        rb3.text = q.options[2]
        rb4.text = q.options[3]
        rgAnswers.clearCheck()
        tvResult.text = ""
        tvResult.visibility = View.GONE
        btnNext.text = getString(R.string.quiz_btn_confirm)
        answered = false
    }

    private fun handleNextButton() {
        if (!answered) {
            val checkedId = rgAnswers.checkedRadioButtonId
            if (checkedId == -1) {
                Toast.makeText(this, getString(R.string.quiz_pick_answer), Toast.LENGTH_SHORT).show()
                return
            }
            answered = true
            val q = QUESTIONS[currentQuestion % QUESTIONS.size]
            val selectedIdx = when (checkedId) {
                R.id.quizRb1 -> 0
                R.id.quizRb2 -> 1
                R.id.quizRb3 -> 2
                else         -> 3
            }
            val correct = selectedIdx == q.correctIdx
            tvResult.text = if (correct)
                getString(R.string.quiz_correct, q.explanation)
            else
                getString(R.string.quiz_wrong, q.options[q.correctIdx])
            tvResult.setTextColor(
                if (correct) ContextCompat.getColor(this, R.color.state_trusted)
                else         ContextCompat.getColor(this, R.color.state_unknown)
            )
            tvResult.visibility = View.VISIBLE
            btnNext.text = getString(R.string.quiz_btn_next)
        } else {
            currentQuestion++
            loadQuestion(currentQuestion)
        }
    }

    private data class QuizQuestion(
        val text: String,
        val options: List<String>,
        val correctIdx: Int,
        val explanation: String = "",
    )

    companion object {
        private val QUESTIONS = listOf(
            QuizQuestion(
                text = "Thủ đô của Việt Nam là gì?",
                options = listOf("Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Huế"),
                correctIdx = 1,
                explanation = "Hà Nội là thủ đô của Việt Nam từ năm 1010."
            ),
            QuizQuestion(
                text = "CPU là viết tắt của?",
                options = listOf("Computer Power Unit", "Central Processing Unit", "Central Program Utility", "Core Processing Unit"),
                correctIdx = 1,
                explanation = "CPU (Central Processing Unit) là bộ xử lý trung tâm của máy tính."
            ),
            QuizQuestion(
                text = "Ngôn ngữ lập trình nào được Google khuyến nghị chính thức cho Android?",
                options = listOf("Java", "C++", "Kotlin", "Python"),
                correctIdx = 2,
                explanation = "Google công bố Kotlin là ngôn ngữ ưu tiên cho Android từ năm 2017."
            ),
            QuizQuestion(
                text = "Machine Learning (Học máy) là?",
                options = listOf("Hệ thống quản lý bộ nhớ", "Giao thức truyền thông mạng", "Phương pháp cho máy tự học từ dữ liệu", "Phần mềm diệt virus"),
                correctIdx = 2,
                explanation = "Machine Learning cho phép hệ thống học từ dữ liệu và cải thiện theo thời gian."
            ),
            QuizQuestion(
                text = "TFLite (TensorFlow Lite) được dùng để làm gì?",
                options = listOf("Lưu trữ đám mây", "Chạy mô hình AI trực tiếp trên thiết bị di động", "Mã hóa dữ liệu mạng", "Quản lý pin thiết bị"),
                correctIdx = 1,
                explanation = "TFLite cho phép triển khai mô hình AI trên thiết bị di động mà không cần kết nối mạng."
            ),
            QuizQuestion(
                text = "Xác thực hành vi sinh trắc học (Behavioural Biometrics) là gì?",
                options = listOf("Quét vân tay khi mở máy", "Nhận dạng qua khuôn mặt", "Phân tích cách người dùng tương tác để nhận dạng liên tục", "Kiểm tra mật khẩu định kỳ"),
                correctIdx = 2,
                explanation = "Xác thực hành vi phân tích các đặc trưng như tốc độ cuộn, nhịp gõ phím, chuyển động IMU để xác thực liên tục."
            ),
        )
    }
}
