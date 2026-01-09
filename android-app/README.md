# PIL Chat - Android Phone App

A modern Android app for interacting with the PIL-VAE (Pseudoinverse Learning - Variational Autoencoder) AI backend.

## Features

- 💬 **Real-time Chat** - Stream responses from the PIL-VAE backend
- 🎓 **Train Knowledge** - Teach the AI new information
- 🎨 **Modern UI** - Material 3 design with dark theme
- 📱 **Phone Optimized** - Designed for regular Android phones (not WearOS)

## Requirements

- Android Studio Hedgehog (2023.1.1) or newer
- Android 8.0 (API 26) or higher
- OnePlus 11R or any modern Android phone
- Same WiFi network as the backend server
- JDK 17

## Quick Start (5 Steps)

### Step 1: Open in Android Studio

1. Open **Android Studio**
2. Click **File → Open**
3. Select: `C:\Users\SANJAY\Desktop\PIL\android-app`
4. Click **OK** and wait for Gradle sync (Android Studio will download gradle automatically)

### Step 2: Update Your PC's IP Address

1. Find your IP: Run `ipconfig` in PowerShell
2. Open `app/build.gradle.kts`
3. Find line with `API_BASE_URL`
4. Replace with your IP: `"http://YOUR_IP:8000"`

### Step 3: Enable USB Debugging on OnePlus 11R

1. **Settings → About Phone**
2. Tap **Build Number** 7 times
3. **Settings → System → Developer Options**
4. Enable **USB Debugging**

### Step 4: Start Backend Server

```powershell
cd C:\Users\SANJAY\Desktop\PIL
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 5: Run the App

1. Connect OnePlus 11R via USB
2. Allow USB debugging when prompted
3. Select your device in Android Studio toolbar
4. Click **Run ▶️** (green play button)

## Project Structure

```
android-app/
├── app/
│   ├── src/main/
│   │   ├── java/com/indxai/pilchat/
│   │   │   ├── PILChatApp.kt          # Application class
│   │   │   ├── data/
│   │   │   │   └── PILApiClient.kt    # API client
│   │   │   ├── di/
│   │   │   │   └── AppModule.kt       # Hilt DI module
│   │   │   └── ui/
│   │   │       ├── MainActivity.kt
│   │   │       ├── Navigation.kt
│   │   │       ├── screens/
│   │   │       │   ├── ChatScreen.kt
│   │   │       │   ├── ChatViewModel.kt
│   │   │       │   ├── SettingsScreen.kt
│   │   │       │   └── TrainScreen.kt
│   │   │       └── theme/
│   │   ├── res/
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
└── settings.gradle.kts
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" | Ensure backend is running with `--host 0.0.0.0` |
| Phone not detected | Enable USB debugging, try different USB cable |
| "Offline" status | Check if phone and PC are on same WiFi |
| Gradle sync fails | File → Invalidate Caches → Restart |
| Build fails | Ensure JDK 17 is installed |

## Tech Stack

- **Kotlin** - Primary language
- **Jetpack Compose** - Modern UI toolkit
- **Material 3** - Design system
- **Hilt** - Dependency injection
- **OkHttp** - Networking
- **Coroutines & Flow** - Async programming

## License

Part of the PIL (Pseudoinverse Learning) project.
