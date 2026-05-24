package com.datn.authenticator

import android.app.Application
import android.util.Log
import com.datn.authenticator.util.NotificationHelper

class AuthenticatorApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "BioAuth Authenticator app starting (versionName=${BuildConfig.VERSION_NAME})")
        NotificationHelper.createChannels(this)
    }

    companion object {
        const val TAG = "BioAuth"
    }
}
