package com.indxai.pilchat.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.indxai.pilchat.ui.theme.PILChatTheme
import dagger.hilt.android.AndroidEntryPoint

/**
 * Main Activity for PIL Chat Android App
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    
    // Permission launcher for audio recording
    private val audioPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Toast.makeText(this, "Voice features enabled", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Voice features require microphone permission", Toast.LENGTH_LONG).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        // Request audio permission if not granted
        checkAndRequestAudioPermission()
        
        setContent {
            PILChatTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    PILChatNavigation()
                }
            }
        }
    }
    
    private fun checkAndRequestAudioPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this, 
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> {
                // Permission already granted
            }
            shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) -> {
                // Show rationale then request
                Toast.makeText(
                    this, 
                    "Microphone permission needed for voice features",
                    Toast.LENGTH_LONG
                ).show()
                audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
            else -> {
                // Request permission directly
                audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }
}
