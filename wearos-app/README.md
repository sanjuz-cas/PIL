# Indxai WearOS Watch App

🎙️ **Voice-First AI Assistant for WearOS** powered by PIL-VAE Engine

![WearOS](https://img.shields.io/badge/WearOS-3.0%2B-blue)
![Kotlin](https://img.shields.io/badge/Kotlin-1.9-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- 🎤 **Voice Input** - Offline speech recognition with Vosk
- 🔊 **Voice Output** - Natural text-to-speech responses  
- 🧠 **PIL-VAE AI** - Gradient-free learning engine backend
- 📱 **Native WearOS** - Jetpack Compose for Wear OS
- 📴 **Offline Ready** - Works without constant connectivity
- 🎨 **Beautiful UI** - Cherry/Beige theme matching web app

## Quick Start

### Prerequisites

1. **Android Studio** Hedgehog (2023.1.1) or newer
2. **JDK 17** (bundled with Android Studio)
3. **WearOS Emulator** or physical watch (Wear OS 3.0+)

### Setup

1. **Open Project in Android Studio**
   ```
   File → Open → Select wearos-app folder
   ```

2. **Download Vosk Model**
   ```powershell
   # Download English model (50MB)
   Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile "vosk-model.zip"
   
   # Extract to assets
   Expand-Archive -Path "vosk-model.zip" -DestinationPath "app\src\main\assets\"
   
   # Rename
   Rename-Item "app\src\main\assets\vosk-model-small-en-us-0.15" "vosk-model"
   ```

3. **Start Backend Server**
   ```bash
   cd ..  # Go to PIL project root
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. **Create WearOS Emulator**
   - Device Manager → Create Device
   - Select "Wear OS Large Round"
   - System Image: Wear OS 4 (API 33)

5. **Run the App**
   - Select emulator from device dropdown
   - Click Run ▶️

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WearOS Watch App                         │
├─────────────────────────────────────────────────────────────┤
│  UI Layer (Jetpack Compose for Wear OS)                    │
│    ├── HomeScreen        - Main voice button               │
│    ├── ListeningScreen   - Active voice capture            │
│    ├── ResponseScreen    - AI response display             │
│    ├── SettingsScreen    - App configuration               │
│    └── HistoryScreen     - Past conversations              │
├─────────────────────────────────────────────────────────────┤
│  Voice Layer                                                │
│    ├── VoskRecognizer    - Offline STT (Vosk)              │
│    └── TTSManager        - Text-to-Speech                  │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│    ├── IndxaiApiClient   - Backend communication           │
│    ├── ChatRepository    - Data orchestration              │
│    └── Room Database     - Local persistence               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS (10.0.2.2:8000)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Indxai Backend (FastAPI + PIL-VAE Engine)                  │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
wearos-app/
├── app/
│   ├── src/main/
│   │   ├── java/com/indxai/watch/
│   │   │   ├── IndxaiWatchApp.kt          # Application class
│   │   │   ├── presentation/
│   │   │   │   ├── MainActivity.kt        # Entry point + navigation
│   │   │   │   ├── screens/               # UI screens
│   │   │   │   ├── viewmodels/            # MVVM ViewModels
│   │   │   │   └── theme/                 # Color theme
│   │   │   ├── voice/
│   │   │   │   ├── VoskRecognizer.kt      # Speech-to-text
│   │   │   │   ├── TTSManager.kt          # Text-to-speech
│   │   │   │   └── VoiceService.kt        # Foreground service
│   │   │   ├── data/
│   │   │   │   ├── api/                   # API client (Ktor)
│   │   │   │   ├── local/                 # Room database
│   │   │   │   └── repository/            # Data repos
│   │   │   └── di/
│   │   │       └── AppModule.kt           # Hilt DI
│   │   ├── res/
│   │   │   └── values/                    # Strings, colors
│   │   ├── assets/
│   │   │   └── vosk-model/                # Voice model (download)
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
├── settings.gradle.kts
└── gradle.properties
```

## Voice Agent: Vosk

We use **Vosk** for offline speech recognition:

- ✅ Works offline (no cloud dependency)
- ✅ Lightweight (50MB model)
- ✅ Fast recognition (<500ms latency)
- ✅ Apache 2.0 License
- ✅ ARM support (WearOS compatible)

### Model Options

| Model | Size | Accuracy | Use Case |
|-------|------|----------|----------|
| vosk-model-small-en-us | 50MB | 85% | **Recommended for WearOS** |
| vosk-model-en-us | 1.8GB | 92% | Desktop/Server |
| vosk-model-small-cn | 42MB | 80% | Chinese |

Download models: https://alphacephei.com/vosk/models

## Configuration

### API Endpoint

Edit `IndxaiApiClient.kt`:
```kotlin
companion object {
    // Emulator uses 10.0.2.2 for host localhost
    const val BASE_URL = "http://10.0.2.2:8000"
    
    // For real device, use your server IP:
    // const val BASE_URL = "http://192.168.1.100:8000"
}
```

### Voice Settings

Adjustable in Settings screen:
- TTS Enabled/Disabled
- Voice Speed (0.75x - 1.5x)
- Auto-speak responses

## Testing

### Emulator Voice Input

1. Click Extended Controls (⋮) in emulator
2. Go to "Microphone" tab
3. Speak into computer microphone

### ADB Commands

```powershell
# View logs
adb logcat | findstr "indxai"

# Install APK
adb install -r app-debug.apk

# Test API connection
adb shell curl http://10.0.2.2:8000/v1/health
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Emulator won't start | Enable Hyper-V/HAXM in BIOS |
| No microphone input | Check emulator audio settings |
| API timeout | Verify backend is running on port 8000 |
| Vosk init failed | Check assets/vosk-model folder exists |
| Gradle sync failed | File → Invalidate Caches → Restart |

## Documentation

- [Full PRD](../docs/PRD_WearOS_VoiceAgent.md)
- [Setup Instructions](../docs/WEAROS_SETUP_INSTRUCTIONS.md)
- [Vosk Documentation](https://alphacephei.com/vosk/)
- [Wear OS Guide](https://developer.android.com/wear)

## License

MIT License - see [LICENSE](../LICENSE)
