package com.datn.authenticator.util

import android.content.Context
import android.util.Log

/**
 * Bối cảnh huấn luyện model, tương ứng 2 checkpoint train sẵn ([WALKING] chỉ đi bộ,
 * [ALL] mọi hoạt động); mỗi mode dùng một sub-folder trong `assets/` và được lưu vào
 * SharedPreferences. Đổi mode phải xoá [com.datn.authenticator.inference.OwnerProfile]
 * và enroll lại vì anchor gắn với embedding của một model cụ thể.
 */
enum class ContextMode(val key: String, val assetFolder: String) {
    WALKING("walking", "walking"),
    ALL("all", "all");

    companion object {
        private const val TAG = "ContextMode"
        private const val PREFS_NAME = "bioauth_prefs"
        private const val KEY_MODE = "context_mode"

        /** Mode mặc định khi user chưa chọn lần nào. */
        val DEFAULT = WALKING

        fun fromKey(k: String?): ContextMode? = entries.firstOrNull { it.key == k }

        /** Đọc mode user đã chọn. Trả về `null` nếu user chưa từng chọn. */
        fun loadSaved(context: Context): ContextMode? {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return fromKey(prefs.getString(KEY_MODE, null))
        }

        /** Đọc mode user đã chọn, fallback về [DEFAULT] nếu chưa chọn. */
        fun loadOrDefault(context: Context): ContextMode =
            loadSaved(context) ?: DEFAULT

        fun save(context: Context, mode: ContextMode) {
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(KEY_MODE, mode.key)
                .apply()
            Log.i(TAG, "Saved context_mode=${mode.key}")
        }

        fun clear(context: Context) {
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .remove(KEY_MODE)
                .apply()
            Log.i(TAG, "Cleared context_mode")
        }

        /**
         * Helper để build path `assets/<mode>/<filename>`.
         * Dùng trong InferenceEngine.
         */
        fun assetPath(mode: ContextMode, filename: String): String =
            "${mode.assetFolder}/$filename"

        /** Kiểm tra mode có đủ file để chạy không (cụ thể là `backbone.tflite`). */
        fun isAvailable(context: Context, mode: ContextMode): Boolean {
            return try {
                context.assets.open(assetPath(mode, "backbone.tflite")).use { /* ok */ }
                true
            } catch (e: Exception) {
                Log.d(TAG, "Mode ${mode.key} not available: ${e.message}")
                false
            }
        }
    }
}
