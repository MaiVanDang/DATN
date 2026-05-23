package com.datn.authenticator.util

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build

object NotificationHelper {
    const val CHANNEL_SERVICE = "bioauth_service"
    const val NOTIFICATION_ID_SERVICE = 1001

    fun createChannels(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        if (manager.getNotificationChannel(CHANNEL_SERVICE) == null) {
            val channel = NotificationChannel(
                CHANNEL_SERVICE,
                "BioAuth Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Ongoing notification while BioAuth is monitoring."
                setShowBadge(false)
            }
            manager.createNotificationChannel(channel)
        }
    }
}
