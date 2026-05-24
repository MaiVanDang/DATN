package com.datn.authenticator.ui

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.datn.authenticator.R
import com.datn.authenticator.fallback.PatternStorage
import com.datn.authenticator.inference.OwnerProfile
import com.datn.authenticator.util.ContextMode
import com.google.android.material.card.MaterialCardView

/**
 * Màn chọn mode train (Walking vs All-action).
 *
 * - Lần đầu mở app: hiện màn này, user chọn mode → vào OwnerEnrollment.
 * - Đã chọn mode + đã enroll: skip thẳng theo flow cũ (Touch/Fallback/Quiz).
 * - User muốn đổi mode sau khi enroll: từ OwnerEnrollment bấm "Đổi mode" →
 *   ContextMode.clear() + OwnerProfile.clear() → quay lại đây.
 */
class ModeSelectActivity : AppCompatActivity() {

    private lateinit var cardWalking: MaterialCardView
    private lateinit var cardAll: MaterialCardView
    private lateinit var tvWalkingHint: TextView
    private lateinit var tvAllHint: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Nếu user đã chọn mode + đã enroll IMU thì bypass màn này
        val savedMode = ContextMode.loadSaved(this)
        val ownerProfile = OwnerProfile(this)
        if (savedMode != null && ownerProfile.hasEnrollment()) {
            routeBasedOnEnrollmentStage(ownerProfile)
            return
        }

        setContentView(R.layout.activity_mode_select)

        cardWalking   = findViewById(R.id.cardWalking)
        cardAll       = findViewById(R.id.cardAll)
        tvWalkingHint = findViewById(R.id.tvWalkingHint)
        tvAllHint     = findViewById(R.id.tvAllHint)

        wireCard(cardWalking, ContextMode.WALKING, tvWalkingHint)
        wireCard(cardAll,     ContextMode.ALL,     tvAllHint)
    }

    /** Một số case cần show preview cho mode đã save trước đó */
    private fun routeBasedOnEnrollmentStage(ownerProfile: OwnerProfile) {
        val intent = when {
            ownerProfile.getRfInertial()?.isTrained == true &&
                PatternStorage(this).isEnrolled() ->
                    Intent(this, QuizActivity::class.java)
            ownerProfile.getRfInertial()?.isTrained == true ->
                    Intent(this, FallbackEnrollActivity::class.java)
            else ->
                    Intent(this, OwnerEnrollmentActivity::class.java)
        }
        startActivity(intent.apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        })
        finish()
    }

    private fun wireCard(card: MaterialCardView, mode: ContextMode, hintView: TextView) {
        val available = ContextMode.isAvailable(this, mode)
        if (!available) {
            card.isEnabled = false
            card.alpha = 0.45f
            hintView.visibility = View.VISIBLE
            hintView.text = getString(R.string.mode_unavailable_hint)
            card.setOnClickListener {
                Toast.makeText(
                    this,
                    getString(R.string.mode_unavailable_toast, mode.key),
                    Toast.LENGTH_LONG
                ).show()
            }
            return
        }
        hintView.visibility = View.GONE
        card.setOnClickListener { confirmAndContinue(mode) }
    }

    private fun confirmAndContinue(mode: ContextMode) {
        // Nếu đã có profile (từ mode khác) — cảnh báo phải xóa
        val ownerProfile = OwnerProfile(this)
        if (ownerProfile.hasEnrollment()) {
            AlertDialog.Builder(this)
                .setTitle(R.string.mode_switch_warning_title)
                .setMessage(getString(R.string.mode_switch_warning_msg, mode.key))
                .setPositiveButton(R.string.mode_switch_confirm) { _, _ ->
                    ownerProfile.clear()
                    PatternStorage(this).clear()
                    persistAndContinue(mode)
                }
                .setNegativeButton(R.string.btn_cancel, null)
                .show()
        } else {
            persistAndContinue(mode)
        }
    }

    private fun persistAndContinue(mode: ContextMode) {
        ContextMode.save(this, mode)
        startActivity(Intent(this, OwnerEnrollmentActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        })
    }
}
