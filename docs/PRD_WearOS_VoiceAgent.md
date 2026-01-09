# Product Requirements Document (PRD)
## Indxai WearOS Watch App with Voice Agent

**Version:** 1.0.0  
**Date:** December 29, 2025  
**Author:** Indxai Team  
**Status:** Draft

---

## 1. Executive Summary

### 1.1 Vision
Transform the Indxai PIL-VAE Engine into a wearable AI assistant accessible via WearOS smartwatches, featuring hands-free voice interaction powered by open-source voice AI technology.

### 1.2 Product Overview
A companion WearOS application that connects to the Indxai backend API, enabling users to:
- Interact with the PIL-VAE AI engine using voice commands
- Receive spoken responses via Text-to-Speech
- Access chat history and knowledge training on-the-go
- Leverage the gradient-free learning capabilities from their wrist

---

## 2. Goals & Objectives

### 2.1 Primary Goals
| Goal | Success Metric |
|------|----------------|
| Voice-first interaction | 90%+ queries via voice input |
| Response latency | < 3 seconds for voice-to-response |
| Battery efficiency | < 5% battery drain per hour of active use |
| Offline capability | Basic commands work without connectivity |

### 2.2 Business Objectives
- Expand Indxai platform to wearable ecosystem
- Demonstrate PIL-VAE edge AI capabilities on resource-constrained devices
- Establish presence in wearable AI assistant market

---

## 3. Target Users

### 3.1 Primary Persona: "Alex the Tech Professional"
- **Demographics:** 25-45, tech-savvy professional
- **Device:** Galaxy Watch 5/6, Pixel Watch, Fossil Gen 6
- **Use Cases:**
  - Quick information queries while hands are busy
  - Email/Gmail checking via voice
  - Knowledge capture during meetings
  - On-the-go AI assistance

### 3.2 Secondary Persona: "Sam the Developer"
- **Demographics:** 20-40, software developer
- **Interests:** Testing edge AI, wearable development
- **Use Cases:**
  - Testing PIL architecture on edge devices
  - Building integrations
  - Research applications

---

## 4. Features & Requirements

### 4.1 Core Features

#### 4.1.1 Voice Input (Speech-to-Text)
**Priority:** P0 (Must Have)

| Requirement | Description |
|-------------|-------------|
| Wake word detection | "Hey Indxai" or customizable |
| Continuous listening | Active listening mode with visual indicator |
| Noise cancellation | Filter background noise for accuracy |
| Multi-language | English primary, extensible to others |

**Recommended Voice Agent: Vosk (Open-Source)**
- Lightweight, offline-capable
- Works on ARM processors (WearOS)
- Apache 2.0 License
- Models as small as 50MB

**Alternative Options:**
| Voice Agent | Pros | Cons | License |
|-------------|------|------|---------|
| **Vosk** | Offline, lightweight, fast | Limited languages | Apache 2.0 |
| **Whisper (OpenAI)** | High accuracy, multilingual | Requires API/large model | MIT |
| **Coqui STT** | Privacy-focused, customizable | Larger models | MPL 2.0 |
| **Chatterbox** | Conversational AI | Heavier resource usage | Open Source |
| **Piper** | Fast TTS companion | STT requires pairing | MIT |

#### 4.1.2 Voice Output (Text-to-Speech)
**Priority:** P0 (Must Have)

| Requirement | Description |
|-------------|-------------|
| Natural voice | Human-like prosody |
| Adjustable speed | 0.5x - 2.0x playback |
| Interrupt support | Stop speaking on new input |
| Low latency | < 500ms to first audio |

**Recommended TTS: Piper (Open-Source)**
- Optimized for edge devices
- Multiple voice options
- MIT License
- ONNX runtime support

#### 4.1.3 Chat Interface
**Priority:** P0 (Must Have)

| Requirement | Description |
|-------------|-------------|
| Circular UI | Optimized for round watch faces |
| Message bubbles | Compact, scrollable history |
| Quick replies | Suggested response chips |
| Haptic feedback | Vibration on message received |

#### 4.1.4 API Integration
**Priority:** P0 (Must Have)

| Endpoint | Function |
|----------|----------|
| `POST /v1/chat` | Send queries, receive streaming responses |
| `POST /v1/train-knowledge` | Train new knowledge from voice |
| `GET /v1/health` | Connection status check |
| `GET /v1/stats` | Engine statistics |

#### 4.1.5 Gmail Integration
**Priority:** P1 (Should Have)

| Feature | Description |
|---------|-------------|
| Voice email check | "Check my emails" command |
| Email summary | Spoken summaries of recent emails |
| Quick actions | Mark read, archive, reply templates |

### 4.2 Secondary Features

#### 4.2.1 Offline Mode
**Priority:** P1 (Should Have)
- Cache recent conversations
- Queue commands for sync
- Basic responses from on-device model

#### 4.2.2 Complications
**Priority:** P2 (Nice to Have)
- Watch face complications showing AI status
- Quick launch shortcuts
- Glanceable information tiles

#### 4.2.3 Fitness Integration
**Priority:** P2 (Nice to Have)
- Health data context for queries
- Workout voice assistance
- Step/activity aware responses

---

## 5. Technical Architecture

### 5.1 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WearOS Watch App                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Vosk      │  │   Piper     │  │  Compose UI Layer   │ │
│  │   (STT)     │  │   (TTS)     │  │  (Wear Compose)     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────▼────────────────▼─────────────────────▼──────────┐ │
│  │              Voice Agent Manager                        │ │
│  │     (Orchestrates STT → API → TTS pipeline)            │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                 │
│  ┌────────────────────────▼───────────────────────────────┐ │
│  │              API Client (Retrofit/Ktor)                 │ │
│  │          Handles /v1/chat, /v1/train-knowledge         │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                 │
│  ┌────────────────────────▼───────────────────────────────┐ │
│  │           Local Cache (Room DB)                         │ │
│  │         Conversation history, offline queue             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Indxai Backend Server                      │
├─────────────────────────────────────────────────────────────┤
│  FastAPI + PIL-VAE Engine + Gmail Tools                     │
│  (Existing infrastructure from PIL project)                 │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Language** | Kotlin | Native Android/WearOS support |
| **UI Framework** | Jetpack Compose for Wear OS | Modern declarative UI |
| **Voice STT** | Vosk | Lightweight, offline-capable |
| **Voice TTS** | Piper / Android TTS | Low latency, customizable |
| **Networking** | Ktor Client | Coroutine-native, lightweight |
| **Local DB** | Room | Standard Android persistence |
| **DI** | Hilt | Recommended for Android |
| **Audio** | AudioRecord + ExoPlayer | Native audio handling |

### 5.3 Voice Agent Integration Details

#### Vosk Integration
```kotlin
// Vosk Model Initialization
val model = Model(assetManager, "vosk-model-small-en-us-0.15")
val recognizer = Recognizer(model, 16000.0f)

// Continuous Recognition
audioRecorder.startRecording()
while (isListening) {
    val buffer = ByteArray(4096)
    audioRecorder.read(buffer, 0, buffer.size)
    if (recognizer.acceptWaveForm(buffer, buffer.size)) {
        val result = recognizer.result
        onTranscription(result)
    }
}
```

#### Piper TTS Integration
```kotlin
// Piper Voice Synthesis
val piper = PiperTTS(context, "en_US-lessac-medium")
val audioData = piper.synthesize("Hello from Indxai")
audioTrack.write(audioData, 0, audioData.size)
audioTrack.play()
```

---

## 6. User Interface Specifications

### 6.1 Screen Designs

#### 6.1.1 Home Screen
```
       ╭──────────────────╮
      ╱                    ╲
     │    🎙️ INDXAI        │
     │                      │
     │   ┌──────────────┐   │
     │   │ Tap to speak │   │
     │   │      🎤      │   │
     │   └──────────────┘   │
     │                      │
     │  ⚙️           📜     │
      ╲                    ╱
       ╰──────────────────╯
```

#### 6.1.2 Listening Screen
```
       ╭──────────────────╮
      ╱                    ╲
     │    🔴 Listening...   │
     │                      │
     │      ∿∿∿∿∿∿∿∿       │
     │    (waveform)        │
     │                      │
     │  "What's the wea..." │
     │                      │
     │        ✕ Cancel      │
      ╲                    ╱
       ╰──────────────────╯
```

#### 6.1.3 Response Screen
```
       ╭──────────────────╮
      ╱                    ╲
     │  🤖 Indxai           │
     │  ─────────────────   │
     │  The weather today   │
     │  is sunny with a     │
     │  high of 72°F...     │
     │                      │
     │  🔊 ⏸️  🎤 New Query  │
      ╲                    ╱
       ╰──────────────────╯
```

### 6.2 Navigation Flow
```
[Home] ──tap──► [Listening] ──result──► [Processing] ──response──► [Response]
   │                                                                    │
   │◄───────────────────────── swipe back ◄─────────────────────────────┤
   │
   ├──swipe left──► [History]
   │
   └──swipe right──► [Settings]
```

### 6.3 Gesture Controls

| Gesture | Action |
|---------|--------|
| Tap center | Start voice input |
| Double tap | Repeat last response |
| Swipe left | View history |
| Swipe right | Settings |
| Long press | Training mode |
| Wrist raise | Wake (optional) |

---

## 7. API Contract

### 7.1 Chat Request
```http
POST /v1/chat
Content-Type: application/x-www-form-urlencoded

query=What+is+the+weather&mode=assistant
```

### 7.2 Chat Response (Streaming NDJSON)
```json
{"type": "chunk", "content": "The weather "}
{"type": "chunk", "content": "today is "}
{"type": "chunk", "content": "sunny."}
{"type": "done", "total_tokens": 15}
```

### 7.3 Health Check
```http
GET /v1/health

Response:
{
  "status": "active",
  "engine": "PIL-VAE Hybrid",
  "version": "1.0.0",
  "memory": {"total_vectors": 1234, "sessions": 56}
}
```

---

## 8. Non-Functional Requirements

### 8.1 Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Voice recognition latency | < 500ms | Time from speech end to text |
| API response time | < 2s | Time from request to first chunk |
| TTS start latency | < 300ms | Time from text to audio start |
| App launch time | < 1.5s | Cold start to interactive |
| Memory usage | < 100MB | Runtime heap allocation |

### 8.2 Reliability
- 99% uptime for API connectivity
- Graceful degradation in offline mode
- Auto-retry with exponential backoff
- Crash-free rate > 99.5%

### 8.3 Security
- HTTPS for all API communication
- OAuth 2.0 for Gmail integration
- No voice data stored on device (privacy mode)
- Secure credential storage (EncryptedSharedPreferences)

### 8.4 Accessibility
- Voice-only operation mode
- High contrast UI option
- Adjustable text size
- Haptic feedback for all interactions

---

## 9. Development Phases

### Phase 1: MVP (4 weeks)
- [ ] Basic WearOS app shell with Compose UI
- [ ] Vosk STT integration
- [ ] API client for /v1/chat
- [ ] Android system TTS (basic)
- [ ] Simple chat history view

### Phase 2: Enhanced Voice (3 weeks)
- [ ] Piper TTS integration
- [ ] Wake word detection
- [ ] Continuous listening mode
- [ ] Streaming response handling

### Phase 3: Full Features (3 weeks)
- [ ] Gmail integration
- [ ] Training mode (voice knowledge input)
- [ ] Watch face complications
- [ ] Offline caching
- [ ] Settings & customization

### Phase 4: Polish (2 weeks)
- [ ] Performance optimization
- [ ] Battery optimization
- [ ] UI/UX refinement
- [ ] Testing & bug fixes

---

## 10. Success Criteria

### 10.1 Launch Criteria
- [ ] All P0 features implemented and tested
- [ ] < 3% crash rate in beta testing
- [ ] Voice recognition accuracy > 90%
- [ ] Average session duration > 2 minutes

### 10.2 Post-Launch KPIs
| KPI | Target | Timeline |
|-----|--------|----------|
| Daily Active Users | 1,000+ | 3 months |
| Session Completion Rate | 80%+ | Launch |
| Voice Query Success Rate | 85%+ | 1 month |
| App Store Rating | 4.0+ | 3 months |

---

## 11. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| WearOS battery drain | High | Medium | Aggressive optimization, limit background activity |
| Voice recognition accuracy in noise | Medium | High | Visual input fallback, noise detection warnings |
| API latency on cellular | Medium | Medium | Response caching, progress indicators |
| Model size for offline STT | Medium | Low | Use smallest Vosk model, streaming download |

---

## 12. Open Questions

1. Should we support Wear OS 3.0+ only or include legacy Wear OS 2.x?
2. Preferred wake word customization approach?
3. Integration with phone companion app vs standalone watch app?
4. Monetization strategy for premium voice features?

---

## 13. Appendix

### A. Competitive Analysis
| App | Voice Support | Offline | Watch Native |
|-----|---------------|---------|--------------|
| Google Assistant | Yes | Partial | Yes |
| Alexa | Yes | No | Limited |
| Siri (watchOS) | Yes | No | Yes |
| **Indxai (Proposed)** | **Yes** | **Yes** | **Yes** |

### B. Voice Model Comparison
| Model | Size | Accuracy | Latency | Offline |
|-------|------|----------|---------|---------|
| Vosk small-en | 50MB | 85% | 200ms | ✅ |
| Vosk large-en | 1.8GB | 92% | 150ms | ✅ |
| Whisper tiny | 75MB | 88% | 300ms | ✅ |
| Google STT | Cloud | 95% | 400ms | ❌ |

### C. References
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [Piper TTS](https://github.com/rhasspy/piper)
- [Wear OS Development Guide](https://developer.android.com/wear)
- [Jetpack Compose for Wear OS](https://developer.android.com/training/wearables/compose)
