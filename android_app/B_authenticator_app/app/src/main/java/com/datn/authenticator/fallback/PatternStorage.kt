package com.datn.authenticator.fallback

import android.content.Context
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * Storage for the user's registered shake pattern (Mục 5.4.1).
 *
 * Per the report:
 *   - The pattern is a small integer (the digit 0–9 that the user picked,
 *     mapped to the raw shake count: digit 1 → 1 shake, …, digit 9 → 9, digit 0 → 10).
 *   - We store both the digit and the verified shake count after registration
 *     (median of 3 enrollment trials) so the verifier doesn't need to recompute.
 *   - Storage uses EncryptedSharedPreferences (AndroidX Security Crypto).
 *     The master key is in Android Keystore (hardware-backed when available).
 *
 * Even though the value being protected is just an integer 1..10, we still
 * encrypt at rest because (a) the report explicitly promises this in §5.4.1,
 * and (b) other secrets may live alongside in this same prefs file later.
 */
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
        // Fallback to plain SharedPreferences if Keystore initialization fails
        // (rare, but happens on some emulators / corrupted keystores).
        Log.w(TAG, "EncryptedSharedPreferences unavailable; falling back to plain prefs. ${e.message}")
        context.getSharedPreferences(PREFS_NAME + "_fallback", Context.MODE_PRIVATE)
    }

    fun isEnrolled(): Boolean = prefs.contains(KEY_SHAKE_COUNT)

    fun savePattern(digit: Int, shakeCount: Int) {
        require(digit in 0..9) { "digit must be 0..9, got $digit" }
        require(shakeCount in 1..20) { "shakeCount must be 1..20, got $shakeCount" }
        prefs.edit().apply {
            putInt(KEY_DIGIT, digit)
            putInt(KEY_SHAKE_COUNT, shakeCount)
            putLong(KEY_ENROLLED_AT, System.currentTimeMillis())
            putInt(KEY_FAILED_ATTEMPTS, 0)
        }.apply()
    }

    /** Returns the registered shake count, or null if not enrolled. */
    fun registeredShakeCount(): Int? =
        if (prefs.contains(KEY_SHAKE_COUNT)) prefs.getInt(KEY_SHAKE_COUNT, -1) else null

    fun registeredDigit(): Int? =
        if (prefs.contains(KEY_DIGIT)) prefs.getInt(KEY_DIGIT, -1) else null

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

    /**
     * Verify [observedCount] against the registered pattern using the ±tolerance
     * rule from Bảng 5.4 (default ±1 shake).
     */
    fun verify(observedCount: Int, tolerance: Int = ShakeDetector.SHAKE_TOLERANCE): Boolean {
        val expected = registeredShakeCount() ?: return false
        return kotlin.math.abs(observedCount - expected) <= tolerance
    }

    companion object {
        private const val TAG = "PatternStorage"
        private const val PREFS_NAME = "shake_pattern_secure"
        private const val KEY_DIGIT = "digit"
        private const val KEY_SHAKE_COUNT = "shake_count"
        private const val KEY_ENROLLED_AT = "enrolled_at"
        private const val KEY_FAILED_ATTEMPTS = "failed_attempts"

        const val MAX_FAILED_ATTEMPTS = 3  // Bảng 5.4 — after this, fall back to system PIN

        /**
         * Map digit 0–9 to expected shake count per the encoding in §5.4.1:
         *   digit 1 → 1 shake, …, digit 9 → 9 shakes, digit 0 → 10 shakes.
         */
        fun digitToShakeCount(digit: Int): Int {
            require(digit in 0..9) { "digit must be 0..9" }
            return if (digit == 0) 10 else digit
        }
    }
}
