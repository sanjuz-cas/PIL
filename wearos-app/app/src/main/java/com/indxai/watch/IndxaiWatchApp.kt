package com.indxai.watch

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * Indxai WearOS Watch Application
 * 
 * A voice-enabled AI assistant powered by PIL-VAE engine.
 * Uses Vosk for speech recognition and Android TTS for voice output.
 */
@HiltAndroidApp
class IndxaiWatchApp : Application() {
    
    override fun onCreate() {
        super.onCreate()
        // Application initialization
    }
}
