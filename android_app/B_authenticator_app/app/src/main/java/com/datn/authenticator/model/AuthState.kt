package com.datn.authenticator.model

/**
 * Authentication state derived from the rolling confidence score S.
 *
 * Mapping per Bảng 5.3 of the thesis report (Mục 5.3.4):
 *   TRUSTED  : S ≥ 0.75   → green lock icon, no action
 *   WARNING  : 0.45 ≤ S < 0.75 → yellow icon, log + prepare fallback
 *   UNKNOWN  : S < 0.45   → red overlay, kick FallbackActivity
 *
 * The thresholds are configurable via SharedPreferences (SettingsActivity)
 * to support thesis §6.4 — different EER thresholds per user.
 */
enum class AuthState {
    TRUSTED,
    WARNING,
    UNKNOWN;

    companion object {
        const val DEFAULT_TRUSTED_THRESHOLD = 0.75f
        const val DEFAULT_WARNING_THRESHOLD = 0.45f

        fun fromScore(
            score: Float,
            trustedThreshold: Float = DEFAULT_TRUSTED_THRESHOLD,
            warningThreshold: Float = DEFAULT_WARNING_THRESHOLD,
        ): AuthState = when {
            score >= trustedThreshold -> TRUSTED
            score >= warningThreshold -> WARNING
            else -> UNKNOWN
        }
    }
}
