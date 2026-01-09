# WearOS Watch App Setup Instructions
## Indxai Voice Assistant for Wear OS

This guide walks you through setting up Android Studio to simulate and develop the Indxai WearOS watch app with integrated voice agent capabilities.

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Android Studio Setup](#2-android-studio-setup)
3. [WearOS Emulator Configuration](#3-wearos-emulator-configuration)
4. [Project Structure](#4-project-structure)
5. [Voice Agent Integration](#5-voice-agent-integration)
6. [Running the App](#6-running-the-app)
7. [Testing Voice Features](#7-testing-voice-features)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### 1.1 System Requirements
| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk Space | 20 GB | 50 GB |
| OS | Windows 10/11, macOS 12+, Linux | Latest stable |
| CPU | x86_64 with virtualization | Intel i7/AMD Ryzen 7+ |

### 1.2 Software Requirements
- **Android Studio**: Hedgehog (2023.1.1) or newer
- **JDK**: 17 or higher (bundled with Android Studio)
- **Kotlin**: 1.9.0+
- **Git**: For version control

### 1.3 Backend Requirements
Ensure the Indxai backend is running:
```bash
# From the PIL project root
cd c:\Users\SANJAY\Desktop\PIL
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. Android Studio Setup

### 2.1 Install Android Studio
1. Download from: https://developer.android.com/studio
2. Run the installer and follow the setup wizard
3. Choose "Standard" installation type

### 2.2 Install Required SDK Components
Open Android Studio → Settings → Languages & Frameworks → Android SDK

**SDK Platforms Tab:**
- [x] Android 14.0 (API 34) - UpsideDownCake
- [x] Android 13.0 (API 33) - Tiramisu
- [x] Wear OS 4 (API 33)
- [x] Wear OS 3 (API 30)

**SDK Tools Tab:**
- [x] Android SDK Build-Tools 34
- [x] Android Emulator
- [x] Android SDK Platform-Tools
- [x] Intel x86 Emulator Accelerator (HAXM) - Windows/Mac
- [x] Android Emulator Hypervisor Driver - Windows

### 2.3 Enable Hardware Acceleration (Critical for Emulator)

**Windows:**
```powershell
# Check if Hyper-V or HAXM is available
# Option 1: Enable HAXM (Intel CPUs)
# Run SDK Manager → SDK Tools → Install HAXM

# Option 2: Enable Windows Hypervisor Platform
# Control Panel → Programs → Turn Windows features on/off
# [x] Windows Hypervisor Platform
# [x] Virtual Machine Platform
```

**Verify:**
```powershell
# Check virtualization is enabled
systeminfo | findstr /i "Hyper-V"
```

---

## 3. WearOS Emulator Configuration

### 3.1 Create WearOS AVD (Android Virtual Device)

1. Open **Device Manager** (Tools → Device Manager)
2. Click **Create Device**
3. Select **Wear OS** category
4. Choose device:
   - **Wear OS Large Round** (recommended for development)
   - **Wear OS Small Round** (for compact testing)
   - **Wear OS Square** (for square watches)

5. Select System Image:
   - **Wear OS 4** (API 33) with Google APIs - **Recommended**
   - Or **Wear OS 3** (API 30) for broader compatibility

6. AVD Configuration:
   ```
   AVD Name: IndxaiWatch_Round
   Startup orientation: Portrait
   Memory: 2048 MB RAM
   VM Heap: 256 MB
   Internal Storage: 2048 MB
   SD Card: 512 MB
   ```

7. Advanced Settings:
   ```
   Graphics: Hardware - GLES 2.0
   Multi-Core CPU: 4 cores
   Boot option: Cold boot
   Keyboard: Enable keyboard input
   ```

### 3.2 Network Configuration for API Access

The emulator needs to reach your local backend server:

```
# Emulator uses 10.0.2.2 to reach host machine's localhost
# Your API URL in the app should be:
http://10.0.2.2:8000/v1/chat
```

### 3.3 Audio Configuration (for Voice Testing)
In AVD settings:
- **Enable Audio Input**: Yes
- **Enable Audio Output**: Yes
- **Microphone**: Host audio (or virtual mic)

---

## 4. Project Structure

### 4.1 Create New WearOS Project

1. File → New → New Project
2. Select **Wear OS** → **Blank Activity (Wear OS)**
3. Configure:
   ```
   Name: IndxaiWatch
   Package name: com.indxai.watch
   Save location: C:\Users\SANJAY\Desktop\PIL\wearos-app
   Language: Kotlin
   Minimum SDK: API 30 (Wear OS 3.0)
   Build configuration: Kotlin DSL (build.gradle.kts)
   ```

### 4.2 Project Directory Structure
```
wearos-app/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/indxai/watch/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── presentation/
│   │   │   │   │   ├── HomeScreen.kt
│   │   │   │   │   ├── ListeningScreen.kt
│   │   │   │   │   ├── ResponseScreen.kt
│   │   │   │   │   ├── SettingsScreen.kt
│   │   │   │   │   └── theme/
│   │   │   │   │       ├── Theme.kt
│   │   │   │   │       └── Color.kt
│   │   │   │   ├── data/
│   │   │   │   │   ├── api/
│   │   │   │   │   │   ├── IndxaiApi.kt
│   │   │   │   │   │   └── ApiClient.kt
│   │   │   │   │   ├── repository/
│   │   │   │   │   │   └── ChatRepository.kt
│   │   │   │   │   └── local/
│   │   │   │   │       ├── ChatDatabase.kt
│   │   │   │   │       └── ChatDao.kt
│   │   │   │   ├── voice/
│   │   │   │   │   ├── VoiceManager.kt
│   │   │   │   │   ├── VoskRecognizer.kt
│   │   │   │   │   └── PiperTTS.kt
│   │   │   │   └── di/
│   │   │   │       └── AppModule.kt
│   │   │   ├── res/
│   │   │   │   ├── values/
│   │   │   │   │   ├── strings.xml
│   │   │   │   │   └── colors.xml
│   │   │   │   └── raw/
│   │   │   │       └── (voice models)
│   │   │   └── AndroidManifest.xml
│   │   └── assets/
│   │       └── vosk-model-small-en-us/
│   │           └── (model files)
│   └── build.gradle.kts
├── gradle/
├── build.gradle.kts
└── settings.gradle.kts
```

### 4.3 Gradle Dependencies (app/build.gradle.kts)

```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
    id("com.google.dagger.hilt.android")
}

android {
    namespace = "com.indxai.watch"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.indxai.watch"
        minSdk = 30
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.8"
    }
}

dependencies {
    // Wear OS Compose
    implementation("androidx.wear.compose:compose-material:1.3.0")
    implementation("androidx.wear.compose:compose-foundation:1.3.0")
    implementation("androidx.wear.compose:compose-navigation:1.3.0")
    
    // Core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.2")
    
    // Compose
    implementation(platform("androidx.compose:compose-bom:2024.01.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    
    // Networking - Ktor
    implementation("io.ktor:ktor-client-android:2.3.7")
    implementation("io.ktor:ktor-client-content-negotiation:2.3.7")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.7")
    
    // Voice Recognition - Vosk
    implementation("com.alphacephei:vosk-android:0.3.47")
    
    // Text-to-Speech - Piper (or Android TTS)
    // implementation("com.github.rhasspy:piper-android:1.0.0")
    
    // Local Database - Room
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")
    
    // Dependency Injection - Hilt
    implementation("com.google.dagger:hilt-android:2.50")
    ksp("com.google.dagger:hilt-compiler:2.50")
    implementation("androidx.hilt:hilt-navigation-compose:1.1.0")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
}
```

---

## 5. Voice Agent Integration

### 5.1 Vosk Speech Recognition Setup

#### Download Voice Model
```powershell
# Create assets directory
mkdir wearos-app\app\src\main\assets\vosk-model

# Download small English model (50MB)
Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile "vosk-model.zip"

# Extract to assets
Expand-Archive -Path "vosk-model.zip" -DestinationPath "wearos-app\app\src\main\assets\"

# Rename folder
Rename-Item "wearos-app\app\src\main\assets\vosk-model-small-en-us-0.15" "vosk-model"
```

#### VoskRecognizer.kt Implementation
```kotlin
package com.indxai.watch.voice

import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class VoskRecognizer @Inject constructor(
    private val context: Context
) {
    private var model: Model? = null
    private var speechService: SpeechService? = null
    
    sealed class RecognitionState {
        object Idle : RecognitionState()
        object Listening : RecognitionState()
        data class PartialResult(val text: String) : RecognitionState()
        data class FinalResult(val text: String) : RecognitionState()
        data class Error(val message: String) : RecognitionState()
    }
    
    suspend fun initialize(): Boolean {
        return try {
            StorageService.unpack(
                context,
                "vosk-model",
                "model",
                { model = Model(it) },
                { throw Exception("Model extraction failed") }
            )
            true
        } catch (e: Exception) {
            false
        }
    }
    
    fun startListening(): Flow<RecognitionState> = flow {
        emit(RecognitionState.Listening)
        
        val recognizer = Recognizer(model, 16000.0f)
        speechService = SpeechService(recognizer, 16000.0f)
        
        speechService?.startListening(object : RecognitionListener {
            override fun onPartialResult(hypothesis: String?) {
                hypothesis?.let { 
                    // Parse JSON result
                    val text = parseVoskResult(it)
                    if (text.isNotEmpty()) {
                        // emit(RecognitionState.PartialResult(text))
                    }
                }
            }
            
            override fun onResult(hypothesis: String?) {
                hypothesis?.let {
                    val text = parseVoskResult(it)
                    if (text.isNotEmpty()) {
                        // emit(RecognitionState.FinalResult(text))
                    }
                }
            }
            
            override fun onFinalResult(hypothesis: String?) {
                hypothesis?.let {
                    val text = parseVoskResult(it)
                    // emit(RecognitionState.FinalResult(text))
                }
            }
            
            override fun onError(exception: Exception?) {
                // emit(RecognitionState.Error(exception?.message ?: "Unknown error"))
            }
            
            override fun onTimeout() {
                // emit(RecognitionState.Idle)
            }
        })
    }
    
    fun stopListening() {
        speechService?.stop()
        speechService?.shutdown()
    }
    
    private fun parseVoskResult(json: String): String {
        // Parse {"text": "hello world"} format
        return try {
            val regex = """"text"\s*:\s*"([^"]*)"""".toRegex()
            regex.find(json)?.groupValues?.get(1) ?: ""
        } catch (e: Exception) {
            ""
        }
    }
}
```

### 5.2 Text-to-Speech Setup

#### Android Native TTS (Recommended for WearOS)
```kotlin
package com.indxai.watch.voice

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.*
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.resume

@Singleton
class TTSManager @Inject constructor(
    private val context: Context
) {
    private var tts: TextToSpeech? = null
    private var isInitialized = false
    
    suspend fun initialize(): Boolean = suspendCancellableCoroutine { cont ->
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                tts?.setSpeechRate(1.0f)
                tts?.setPitch(1.0f)
                isInitialized = true
                cont.resume(true)
            } else {
                cont.resume(false)
            }
        }
    }
    
    suspend fun speak(text: String): Boolean = suspendCancellableCoroutine { cont ->
        if (!isInitialized) {
            cont.resume(false)
            return@suspendCancellableCoroutine
        }
        
        val utteranceId = UUID.randomUUID().toString()
        
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            
            override fun onDone(utteranceId: String?) {
                cont.resume(true)
            }
            
            override fun onError(utteranceId: String?) {
                cont.resume(false)
            }
        })
        
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }
    
    fun stop() {
        tts?.stop()
    }
    
    fun shutdown() {
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
```

### 5.3 Piper TTS Alternative (Higher Quality)

For higher quality voice synthesis, integrate Piper:

```kotlin
// Add to build.gradle.kts (when Piper Android library is available)
// implementation("com.github.rhasspy:piper-android:1.0.0")

// Alternative: Use Piper via JNI/NDK
// Download ONNX model from: https://github.com/rhasspy/piper/releases

class PiperTTS(private val context: Context) {
    // Native library loading
    init {
        System.loadLibrary("piper_jni")
    }
    
    external fun initModel(modelPath: String): Long
    external fun synthesize(handle: Long, text: String): ShortArray
    external fun destroy(handle: Long)
}
```

---

## 6. Running the App

### 6.1 Start the Emulator

**Method 1: From Android Studio**
1. Open Device Manager (View → Tool Windows → Device Manager)
2. Click ▶️ next to "IndxaiWatch_Round"
3. Wait for boot (~2-3 minutes first time)

**Method 2: From Command Line**
```powershell
# List available AVDs
$env:ANDROID_HOME\emulator\emulator -list-avds

# Start specific AVD
$env:ANDROID_HOME\emulator\emulator -avd IndxaiWatch_Round -gpu host
```

### 6.2 Deploy the App

**From Android Studio:**
1. Select "IndxaiWatch_Round" from device dropdown
2. Click Run ▶️ (Shift+F10)
3. Wait for Gradle build and installation

**From Command Line:**
```powershell
cd wearos-app
.\gradlew installDebug

# Or with ADB
adb -s emulator-5554 install app\build\outputs\apk\debug\app-debug.apk
```

### 6.3 Verify Backend Connection

```powershell
# Make sure backend is running
curl http://localhost:8000/v1/health

# From emulator, test connection
adb shell curl http://10.0.2.2:8000/v1/health
```

---

## 7. Testing Voice Features

### 7.1 Emulator Microphone Setup

**Option 1: Use Host Microphone**
1. In emulator settings → Microphone → Virtual microphone uses host audio
2. Speak into your computer's microphone

**Option 2: Push Audio File**
```powershell
# Record a test audio file
# Push to emulator
adb push test_query.wav /sdcard/test_query.wav

# Play back during voice capture (testing purposes)
```

**Option 3: Extended Controls**
1. Click ⋮ in emulator toolbar → Extended Controls
2. Go to "Microphone" tab
3. Use "Virtual Mic" to inject audio

### 7.2 Test Voice Recognition

```kotlin
// Test activity to verify Vosk
class VoiceTestActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        lifecycleScope.launch {
            val recognizer = VoskRecognizer(applicationContext)
            
            if (recognizer.initialize()) {
                Log.d("VoiceTest", "Vosk initialized successfully")
                
                recognizer.startListening().collect { state ->
                    when (state) {
                        is VoskRecognizer.RecognitionState.FinalResult -> {
                            Log.d("VoiceTest", "Recognized: ${state.text}")
                        }
                        else -> {}
                    }
                }
            } else {
                Log.e("VoiceTest", "Vosk initialization failed")
            }
        }
    }
}
```

### 7.3 Test API Integration

```kotlin
// Test API call from watch
suspend fun testApiCall() {
    val client = HttpClient(Android) {
        install(ContentNegotiation) {
            json()
        }
    }
    
    val response: HttpResponse = client.submitForm(
        url = "http://10.0.2.2:8000/v1/chat",
        formParameters = parameters {
            append("query", "Hello from WearOS!")
            append("mode", "assistant")
        }
    )
    
    // Handle streaming response
    response.bodyAsChannel().readRemaining().readText()
}
```

---

## 8. Troubleshooting

### 8.1 Common Issues

| Issue | Solution |
|-------|----------|
| Emulator won't start | Enable Hyper-V/HAXM, check BIOS virtualization |
| No audio input | Check emulator audio settings, use Extended Controls |
| API connection failed | Use `10.0.2.2` instead of `localhost`, check firewall |
| Vosk model not found | Verify assets folder structure, check file size |
| Out of memory | Increase AVD RAM, use smaller voice model |
| Slow emulator | Use Cold Boot, enable GPU acceleration |

### 8.2 ADB Commands

```powershell
# List connected devices
adb devices

# Install APK
adb install -r app-debug.apk

# View logs
adb logcat | findstr "Indxai"

# Shell access
adb shell

# Port forwarding (if needed)
adb reverse tcp:8000 tcp:8000

# Take screenshot
adb shell screencap -p /sdcard/screen.png
adb pull /sdcard/screen.png

# Record screen
adb shell screenrecord /sdcard/demo.mp4
```

### 8.3 Debug Voice Recognition

```powershell
# Check microphone permissions
adb shell dumpsys package com.indxai.watch | findstr "RECORD_AUDIO"

# Monitor audio input
adb shell dumpsys audio

# Check Vosk logs
adb logcat | findstr "vosk"
```

### 8.4 Performance Profiling

```powershell
# CPU profiling
adb shell top -n 1 | findstr "indxai"

# Memory usage
adb shell dumpsys meminfo com.indxai.watch

# Battery impact
adb shell dumpsys batterystats | findstr "indxai"
```

---

## Next Steps

1. **Build the Project**: Follow Section 4 to create the Android Studio project
2. **Add Voice Models**: Download and configure Vosk models (Section 5.1)
3. **Implement UI**: Build the Wear Compose screens
4. **Test Locally**: Run in emulator and verify all features
5. **Deploy to Device**: Test on real WearOS hardware

---

## Resources

- [Wear OS Developer Guide](https://developer.android.com/wear)
- [Jetpack Compose for Wear OS](https://developer.android.com/training/wearables/compose)
- [Vosk Documentation](https://alphacephei.com/vosk/)
- [Piper TTS GitHub](https://github.com/rhasspy/piper)
- [Ktor Client](https://ktor.io/docs/getting-started-ktor-client.html)
- [Android Studio Emulator](https://developer.android.com/studio/run/emulator)
