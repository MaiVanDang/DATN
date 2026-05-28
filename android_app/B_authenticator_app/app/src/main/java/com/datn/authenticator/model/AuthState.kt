package com.datn.authenticator.model

enum class AuthState {
    TRUSTED,
    WARNING,
    UNKNOWN;

    companion object {
        const val DEFAULT_TRUSTED_THRESHOLD = 0.5f
        const val DEFAULT_WARNING_THRESHOLD = 0.3f

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
