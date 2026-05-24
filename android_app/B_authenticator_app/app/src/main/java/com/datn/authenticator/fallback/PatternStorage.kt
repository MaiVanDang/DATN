package com.datn.authenticator.fallback

import android.content.Context
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import kotlin.math.abs

class PatternStorage(context: Context) {
    private val prefs: android.content.SharedPreferences = try {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context,
            PREFS_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    } catch (e: Exception) {
        Log.w(TAG, "EncryptedSharedPreferences unavailable; falling back to plain prefs. ${e.message}")
        context.getSharedPreferences(PREFS_NAME + "_fallback", Context.MODE_PRIVATE)
    }

    fun isEnrolled(): Boolean = prefs.contains(KEY_SHAKE_COUNT)

    fun savePattern(shakeCount: Int) {
        require(shakeCount in 1..20) { "shakeCount must be 1..20, got $shakeCount" }
        prefs.edit().apply {
            putInt(KEY_SHAKE_COUNT, shakeCount)
            putLong(KEY_ENROLLED_AT, System.currentTimeMillis())
            putInt(KEY_FAILED_ATTEMPTS, 0)
        }.apply()
    }

    fun registeredShakeCount(): Int? =
        if (prefs.contains(KEY_SHAKE_COUNT)) prefs.getInt(KEY_SHAKE_COUNT, -1) else null

    fun failedAttempts(): Int = prefs.getInt(KEY_FAILED_ATTEMPTS, 0)

    fun recordFailedAttempt(): Int {
        val n = failedAttempts() + 1
        prefs.edit().putInt(KEY_FAILED_ATTEMPTS, n).apply()
        return n
    }

    fun resetFailedAttempts() {
        prefs.edit().putInt(KEY_FAILED_ATTEMPTS, 0).apply()
    }

    fun clear() {
        prefs.edit().clear().apply()
    }

    fun verify(observedCount: Int, tolerance: Int = ShakeDetector.SHAKE_TOLERANCE): Boolean {
        val expected = registeredShakeCount() ?: return false
        return abs(observedCount - expected) <= tolerance
    }

    companion object {
        private const val TAG = "PatternStorage"
        private const val PREFS_NAME = "shake_pattern_secure"
        private const val KEY_SHAKE_COUNT = "shake_count"
        private const val KEY_ENROLLED_AT = "enrolled_at"
        private const val KEY_FAILED_ATTEMPTS = "failed_attempts"

        const val MAX_FAILED_ATTEMPTS = 3
    }
}
