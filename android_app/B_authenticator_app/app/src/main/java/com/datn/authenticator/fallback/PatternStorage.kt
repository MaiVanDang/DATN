package com.datn.authenticator.fallback

import android.content.Context
import android.util.Base64
import android.util.Log
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import java.security.MessageDigest
import java.security.SecureRandom

/**
 * Lưu "mật khẩu lắc" dạng dãy chữ số (mỗi chữ số 1..9, độ dài 4..8): mỗi chữ số là
 * số lần lắc trong một cụm, cách nhau bằng quãng dừng (xem [ShakeDetector]).
 * Không lưu dãy thô — lưu salt ngẫu nhiên + SHA-256(salt || dãy), verify bằng băm lại.
 */
class PatternStorage(context: Context) {

    private val prefs = try {
        val key = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build()
        EncryptedSharedPreferences.create(
            context, PREFS_NAME, key,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    } catch (e: Exception) {
        Log.w(TAG, "EncryptedSharedPreferences unavailable: ${e.message}")
        context.getSharedPreferences(PREFS_NAME + "_fallback", Context.MODE_PRIVATE)
    }

    fun isEnrolled(): Boolean = prefs.contains(KEY_HASH) && prefs.contains(KEY_SALT)

    /** Lưu dãy chữ số. Ném lỗi nếu độ dài hoặc chữ số không hợp lệ. */
    fun savePattern(sequence: List<Int>) {
        validate(sequence)
        val salt = ByteArray(SALT_LEN).also { SecureRandom().nextBytes(it) }
        val hash = hash(salt, sequence)
        prefs.edit()
            .putString(KEY_SALT, Base64.encodeToString(salt, Base64.NO_WRAP))
            .putString(KEY_HASH, Base64.encodeToString(hash, Base64.NO_WRAP))
            .putInt(KEY_LEN, sequence.size)
            .putLong(KEY_ENROLLED_AT, System.currentTimeMillis())
            .putInt(KEY_FAILED_ATTEMPTS, 0)
            .apply()
    }

    /** Độ dài dãy đã đăng ký (để UI hiển thị "cần nhập N chữ số"). */
    fun registeredLength(): Int = prefs.getInt(KEY_LEN, 0)

    fun verify(observed: List<Int>): Boolean {
        if (!isEnrolled()) return false
        if (observed.size != registeredLength()) return false
        val salt = Base64.decode(prefs.getString(KEY_SALT, null) ?: return false, Base64.NO_WRAP)
        val expected = Base64.decode(prefs.getString(KEY_HASH, null) ?: return false, Base64.NO_WRAP)
        return MessageDigest.isEqual(hash(salt, observed), expected)
    }

    fun failedAttempts(): Int = prefs.getInt(KEY_FAILED_ATTEMPTS, 0)
    fun recordFailedAttempt(): Int =
        (failedAttempts() + 1).also { prefs.edit().putInt(KEY_FAILED_ATTEMPTS, it).apply() }
    fun resetFailedAttempts() = prefs.edit().putInt(KEY_FAILED_ATTEMPTS, 0).apply()
    fun clear() = prefs.edit().clear().apply()

    private fun hash(salt: ByteArray, seq: List<Int>): ByteArray {
        val md = MessageDigest.getInstance("SHA-256")
        md.update(salt)
        md.update(seq.joinToString(",").toByteArray(Charsets.UTF_8))
        return md.digest()
    }

    companion object {
        private const val TAG = "PatternStorage"
        private const val PREFS_NAME = "shake_pattern_secure"
        private const val KEY_SALT = "salt"
        private const val KEY_HASH = "hash"
        private const val KEY_LEN  = "seq_len"
        private const val KEY_ENROLLED_AT = "enrolled_at"
        private const val KEY_FAILED_ATTEMPTS = "failed_attempts"
        private const val SALT_LEN = 16

        const val MAX_FAILED_ATTEMPTS = 3
        const val MIN_DIGITS = 4
        const val MAX_DIGITS = 8
        const val MIN_DIGIT_VALUE = 1
        const val MAX_DIGIT_VALUE = 9

        fun validate(seq: List<Int>) {
            require(seq.size in MIN_DIGITS..MAX_DIGITS) {
                "Độ dài dãy phải $MIN_DIGITS..$MAX_DIGITS, nhận ${seq.size}"
            }
            require(seq.all { it in MIN_DIGIT_VALUE..MAX_DIGIT_VALUE }) {
                "Mỗi chữ số phải $MIN_DIGIT_VALUE..$MAX_DIGIT_VALUE"
            }
        }
    }
}
