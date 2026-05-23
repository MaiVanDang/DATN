package com.datn.authenticator.util

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.util.Log
import com.datn.authenticator.service.AuthenticationService

/**
 * Auto-starts the AuthenticationService after device boot if the user
 * previously enabled the "start on boot" preference.
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action ?: return
        if (action != Intent.ACTION_BOOT_COMPLETED &&
            action != "android.intent.action.QUICKBOOT_POWERON") return

        val prefs: SharedPreferences =
            context.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val autoStart = prefs.getBoolean("auto_start_on_boot", true)

        if (autoStart) {
            Log.i(TAG, "Boot completed — auto-starting AuthenticationService")
            AuthenticationService.start(context)
        } else {
            Log.d(TAG, "Boot completed but auto_start_on_boot=false; skipping")
        }
    }

    companion object {
        private const val TAG = "BootReceiver"
    }
}
