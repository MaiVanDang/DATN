package com.datn.authenticator.util

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.util.Log
import com.datn.authenticator.fallback.PatternStorage
import com.datn.authenticator.inference.OwnerProfile
import com.datn.authenticator.service.AuthenticationService

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action ?: return
        if (action != Intent.ACTION_BOOT_COMPLETED &&
            action != "android.intent.action.QUICKBOOT_POWERON") return

        val prefs: SharedPreferences =
            context.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val autoStart = prefs.getBoolean("auto_start_on_boot", true)
        if (!autoStart) {
            Log.d(TAG, "Boot completed but auto_start_on_boot=false; skipping")
            return
        }

        val profile = OwnerProfile(context)
        val fullyEnrolled = profile.hasEnrollment() &&
            profile.getRfInertial()?.isTrained == true &&
            PatternStorage(context).isEnrolled()
        if (!fullyEnrolled) {
            Log.d(TAG, "Boot completed but user not fully enrolled; skipping auto-start")
            return
        }

        Log.i(TAG, "Boot completed — auto-starting AuthenticationService")
        AuthenticationService.start(context)
    }

    companion object {
        private const val TAG = "BootReceiver"
    }
}
