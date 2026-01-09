package com.indxai.pilchat

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * PIL Chat Android Application
 * Main Application class with Hilt dependency injection
 */
@HiltAndroidApp
class PILChatApp : Application() {
    
    override fun onCreate() {
        super.onCreate()
        // Initialize any app-wide configurations here
    }
}
