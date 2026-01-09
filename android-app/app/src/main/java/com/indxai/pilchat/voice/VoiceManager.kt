package com.indxai.pilchat.voice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.indxai.pilchat.util.TextCleaner
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Voice state for UI
 */
sealed class VoiceState {
    object Idle : VoiceState()
    object WaitingForWakeWord : VoiceState()
    object Listening : VoiceState()
    object Processing : VoiceState()
    object Speaking : VoiceState()
    data class Error(val message: String) : VoiceState()
}

/**
 * Voice command types
 */
sealed class VoiceCommand {
    object None : VoiceCommand()
    object StopSpeaking : VoiceCommand()
    object ClearChat : VoiceCommand()
    object RepeatLastResponse : VoiceCommand()
    object EnableContinuousMode : VoiceCommand()
    object DisableContinuousMode : VoiceCommand()
    object CheckStatus : VoiceCommand()
    data class Query(val text: String) : VoiceCommand()
}

/**
 * Voice Intelligence Manager
 * 
 * Features:
 * - Wake word detection ("Hey PIL", "OK Assistant")
 * - Voice commands (stop, clear, repeat, etc.)
 * - Continuous conversation mode
 * - Sentiment-aware TTS speed
 * - Haptic feedback
 * - Output cleaning for TTS
 */
@Singleton
class VoiceManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        private const val TAG = "VoiceIntelligence"
        
        // Wake words (case insensitive)
        val WAKE_WORDS = listOf(
            "hey pill", "hey pil", "hey p i l", "hey pi l",
            "ok pill", "okay pill", "ok pil", "okay pil",
            "hey assistant", "ok assistant", "hello assistant",
            "hi pill", "hi pil"
        )
        
        // Stop commands
        val STOP_COMMANDS = listOf(
            "stop", "stop talking", "shut up", "be quiet",
            "cancel", "nevermind", "never mind", "quiet",
            "silence", "enough", "stop it", "stop speaking"
        )
        
        // Clear chat commands  
        val CLEAR_COMMANDS = listOf(
            "clear chat", "clear history", "start over",
            "new conversation", "reset", "clear everything",
            "delete history", "fresh start", "start fresh"
        )
        
        // Repeat commands
        val REPEAT_COMMANDS = listOf(
            "repeat", "say that again", "repeat that",
            "what did you say", "pardon", "come again",
            "repeat last", "say it again", "one more time"
        )
        
        // Status check commands
        val STATUS_COMMANDS = listOf(
            "status", "are you there", "hello", "hey",
            "are you online", "check connection", "ping"
        )
        
        // Continuous mode commands
        val CONTINUOUS_ON_COMMANDS = listOf(
            "continuous mode", "keep listening", "conversation mode",
            "always listen", "stay on", "hands free mode"
        )
        
        val CONTINUOUS_OFF_COMMANDS = listOf(
            "stop listening", "disable continuous", "turn off listening",
            "normal mode", "manual mode", "stop continuous"
        )
    }
    
    // State flows
    private val _voiceState = MutableStateFlow<VoiceState>(VoiceState.Idle)
    val voiceState: StateFlow<VoiceState> = _voiceState.asStateFlow()
    
    private val _recognizedText = MutableStateFlow("")
    val recognizedText: StateFlow<String> = _recognizedText.asStateFlow()
    
    private val _partialText = MutableStateFlow("")
    val partialText: StateFlow<String> = _partialText.asStateFlow()
    
    private val _isTTSEnabled = MutableStateFlow(true)
    val isTTSEnabled: StateFlow<Boolean> = _isTTSEnabled.asStateFlow()
    
    private val _isContinuousModeEnabled = MutableStateFlow(false)
    val isContinuousModeEnabled: StateFlow<Boolean> = _isContinuousModeEnabled.asStateFlow()
    
    private val _isWakeWordEnabled = MutableStateFlow(true)
    val isWakeWordEnabled: StateFlow<Boolean> = _isWakeWordEnabled.asStateFlow()
    
    private val _lastResponse = MutableStateFlow("")
    val lastResponse: StateFlow<String> = _lastResponse.asStateFlow()
    
    // Speech Recognizer
    private var speechRecognizer: SpeechRecognizer? = null
    private var isRecognizerAvailable = false
    
    // Text-to-Speech
    private var textToSpeech: TextToSpeech? = null
    private var isTTSReady = false
    
    // Vibrator for haptic feedback
    private val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    
    // Callbacks
    private var onQueryCallback: ((String) -> Unit)? = null
    private var onCommandCallback: ((VoiceCommand) -> Unit)? = null
    
    // Mode flags
    private var isWakeWordMode = false
    
    init {
        initializeSpeechRecognizer()
        initializeTextToSpeech()
    }
    
    private fun initializeSpeechRecognizer() {
        isRecognizerAvailable = SpeechRecognizer.isRecognitionAvailable(context)
        
        if (isRecognizerAvailable) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
                setRecognitionListener(createRecognitionListener())
            }
            Log.d(TAG, "Speech recognizer initialized")
        } else {
            Log.w(TAG, "Speech recognition not available")
        }
    }
    
    /**
     * Recreate speech recognizer to recover from errors
     */
    private fun recreateSpeechRecognizer() {
        try {
            speechRecognizer?.destroy()
            speechRecognizer = null
            
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                initializeSpeechRecognizer()
                Log.d(TAG, "Speech recognizer recreated")
            }, 200)
        } catch (e: Exception) {
            Log.e(TAG, "Error recreating speech recognizer", e)
        }
    }
    
    private fun initializeTextToSpeech() {
        textToSpeech = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech?.setLanguage(Locale.US)
                isTTSReady = result != TextToSpeech.LANG_MISSING_DATA && 
                             result != TextToSpeech.LANG_NOT_SUPPORTED
                
                textToSpeech?.setSpeechRate(1.0f)
                textToSpeech?.setPitch(1.0f)
                
                textToSpeech?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {
                        _voiceState.value = VoiceState.Speaking
                    }
                    
                    override fun onDone(utteranceId: String?) {
                        _voiceState.value = VoiceState.Idle
                        
                        // Resume listening in continuous mode
                        if (_isContinuousModeEnabled.value) {
                            startListeningForWakeWord()
                        }
                    }
                    
                    override fun onError(utteranceId: String?) {
                        _voiceState.value = VoiceState.Error("TTS Error")
                    }
                })
                
                Log.d(TAG, "TTS initialized")
            }
        }
    }
    
    private fun createRecognitionListener() = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            Log.d(TAG, "Ready for speech")
            _voiceState.value = if (isWakeWordMode) VoiceState.WaitingForWakeWord else VoiceState.Listening
            _partialText.value = ""
            vibrateShort()
        }
        
        override fun onBeginningOfSpeech() {
            Log.d(TAG, "Speech started")
        }
        
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        
        override fun onEndOfSpeech() {
            Log.d(TAG, "Speech ended")
            _voiceState.value = VoiceState.Processing
        }
        
        override fun onError(error: Int) {
            val errorMessage = getErrorMessage(error)
            Log.e(TAG, "Recognition error: $errorMessage (code: $error)")
            
            _partialText.value = ""
            
            when (error) {
                SpeechRecognizer.ERROR_NO_MATCH -> {
                    // No speech detected - silently retry in continuous mode
                    _voiceState.value = VoiceState.Idle
                    if (_isContinuousModeEnabled.value || isWakeWordMode) {
                        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                            startListeningForWakeWord()
                        }, 300)
                    }
                }
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> {
                    // Speech timeout - normal, just reset
                    _voiceState.value = VoiceState.Idle
                    if (_isContinuousModeEnabled.value) {
                        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                            startListeningForWakeWord()
                        }, 300)
                    }
                }
                SpeechRecognizer.ERROR_CLIENT -> {
                    // Client error - recreate recognizer and retry
                    Log.w(TAG, "Client error - recreating speech recognizer")
                    _voiceState.value = VoiceState.Idle
                    recreateSpeechRecognizer()
                }
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> {
                    // Recognizer busy - wait and retry
                    _voiceState.value = VoiceState.Idle
                    android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                        if (_isContinuousModeEnabled.value) {
                            startListeningForWakeWord()
                        }
                    }, 500)
                }
                else -> {
                    // Show error for other cases but don't persist
                    _voiceState.value = VoiceState.Error(errorMessage)
                    android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                        _voiceState.value = VoiceState.Idle
                    }, 2000)
                }
            }
        }
        
        override fun onResults(results: Bundle?) {
            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val text = matches?.firstOrNull()?.lowercase() ?: ""
            
            Log.d(TAG, "Recognized: $text")
            _partialText.value = ""
            
            if (isWakeWordMode) {
                handleWakeWordResult(text)
            } else {
                handleSpeechResult(text)
            }
        }
        
        override fun onPartialResults(partialResults: Bundle?) {
            val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val partial = matches?.firstOrNull() ?: ""
            _partialText.value = partial
        }
        
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }
    
    private fun handleWakeWordResult(text: String) {
        // Check if wake word was detected
        val wakeWordDetected = WAKE_WORDS.any { text.contains(it) }
        
        if (wakeWordDetected) {
            Log.d(TAG, "Wake word detected!")
            vibrateConfirm()
            
            // Extract command after wake word
            var command = text
            WAKE_WORDS.forEach { wakeWord ->
                command = command.replace(wakeWord, "").trim()
            }
            
            if (command.isNotBlank()) {
                // Wake word + command in same utterance
                handleSpeechResult(command)
            } else {
                // Just wake word, start listening for command
                isWakeWordMode = false
                startListeningInternal()
            }
        } else {
            // No wake word, continue listening in continuous mode
            if (_isContinuousModeEnabled.value) {
                startListeningForWakeWord()
            } else {
                _voiceState.value = VoiceState.Idle
            }
        }
    }
    
    private fun handleSpeechResult(text: String) {
        Log.d(TAG, "handleSpeechResult: '$text'")
        
        if (text.isBlank()) {
            Log.d(TAG, "Empty speech result, returning to Idle")
            _voiceState.value = VoiceState.Idle
            return
        }
        
        // Check for voice commands
        val command = parseVoiceCommand(text)
        Log.d(TAG, "Parsed command: $command")
        
        when (command) {
            is VoiceCommand.Query -> {
                Log.d(TAG, "Invoking query callback with: ${command.text}")
                _recognizedText.value = command.text
                onQueryCallback?.invoke(command.text)
                    ?: Log.e(TAG, "ERROR: onQueryCallback is NULL!")
            }
            else -> {
                Log.d(TAG, "Invoking command callback")
                onCommandCallback?.invoke(command)
            }
        }
        
        _voiceState.value = VoiceState.Idle
        
        // Resume wake word listening in continuous mode
        if (_isContinuousModeEnabled.value) {
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                startListeningForWakeWord()
            }, 500)
        }
    }
    
    private fun parseVoiceCommand(text: String): VoiceCommand {
        val lowercaseText = text.lowercase().trim()
        
        return when {
            STOP_COMMANDS.any { lowercaseText.contains(it) } -> VoiceCommand.StopSpeaking
            CLEAR_COMMANDS.any { lowercaseText.contains(it) } -> VoiceCommand.ClearChat
            REPEAT_COMMANDS.any { lowercaseText.contains(it) } -> VoiceCommand.RepeatLastResponse
            STATUS_COMMANDS.any { lowercaseText == it } -> VoiceCommand.CheckStatus
            CONTINUOUS_ON_COMMANDS.any { lowercaseText.contains(it) } -> VoiceCommand.EnableContinuousMode
            CONTINUOUS_OFF_COMMANDS.any { lowercaseText.contains(it) } -> VoiceCommand.DisableContinuousMode
            else -> VoiceCommand.Query(text)
        }
    }
    
    /**
     * Start listening with wake word detection
     */
    fun startListeningForWakeWord() {
        if (!isRecognizerAvailable) return
        
        isWakeWordMode = true
        startListeningInternal()
    }
    
    /**
     * Start listening without wake word (direct input)
     */
    fun startListening(
        onQuery: (String) -> Unit,
        onCommand: ((VoiceCommand) -> Unit)? = null
    ) {
        Log.d(TAG, "startListening called - isRecognizerAvailable: $isRecognizerAvailable, speechRecognizer: ${speechRecognizer != null}")
        
        if (!isRecognizerAvailable || speechRecognizer == null) {
            _voiceState.value = VoiceState.Error("Speech recognition not available")
            Log.e(TAG, "Speech recognizer not available, attempting to recreate")
            // Try to reinitialize
            recreateSpeechRecognizer()
            return
        }
        
        // Stop any ongoing recognition first
        try {
            speechRecognizer?.cancel()
        } catch (e: Exception) {
            Log.e(TAG, "Error canceling previous recognition", e)
        }
        
        stopSpeaking()
        
        onQueryCallback = onQuery
        onCommandCallback = onCommand
        isWakeWordMode = false
        
        Log.d(TAG, "Setting up direct listening (no wake word)")
        _voiceState.value = VoiceState.Listening
        
        // Ensure we're on main thread and add small delay for stability
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            startListeningInternal()
        }, 150)
    }
    
    private fun startListeningInternal() {
        if (speechRecognizer == null) {
            Log.e(TAG, "Speech recognizer is null")
            _voiceState.value = VoiceState.Error("Recognizer not ready")
            return
        }
        
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            
            // Longer timeout for wake word mode
            val silenceTimeout = if (isWakeWordMode) 3000L else 2000L
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, silenceTimeout)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, silenceTimeout)
        }
        
        try {
            speechRecognizer?.startListening(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recognition", e)
            _voiceState.value = VoiceState.Error("Failed to start listening")
            // Try to recover
            recreateSpeechRecognizer()
        }
    }
    
    /**
     * Stop listening
     */
    fun stopListening() {
        try {
            speechRecognizer?.stopListening()
            speechRecognizer?.cancel()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping listener", e)
        }
        _voiceState.value = VoiceState.Idle
        _partialText.value = ""
        isWakeWordMode = false
    }
    
    /**
     * Speak text using TTS (cleans output first)
     */
    fun speak(text: String, saveAsLastResponse: Boolean = true) {
        if (!isTTSReady || !_isTTSEnabled.value) return
        
        // Clean text for TTS
        val cleanText = TextCleaner.cleanForTTS(text)
        if (cleanText.isBlank()) return
        
        if (saveAsLastResponse) {
            _lastResponse.value = text
        }
        
        // Adjust speech rate based on text length
        val rate = when {
            cleanText.length > 500 -> 1.1f  // Faster for long text
            cleanText.length < 50 -> 0.95f  // Slightly slower for short
            else -> 1.0f
        }
        textToSpeech?.setSpeechRate(rate)
        
        val params = Bundle().apply {
            putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "pil_${System.currentTimeMillis()}")
        }
        
        textToSpeech?.speak(cleanText, TextToSpeech.QUEUE_FLUSH, params, "pil_response")
        Log.d(TAG, "Speaking: ${cleanText.take(50)}...")
    }
    
    /**
     * Repeat last response
     */
    fun repeatLastResponse() {
        if (_lastResponse.value.isNotBlank()) {
            speak(_lastResponse.value, saveAsLastResponse = false)
        }
    }
    
    /**
     * Stop speaking
     */
    fun stopSpeaking() {
        if (textToSpeech?.isSpeaking == true) {
            textToSpeech?.stop()
            _voiceState.value = VoiceState.Idle
        }
    }
    
    /**
     * Toggle TTS
     */
    fun toggleTTS() {
        _isTTSEnabled.value = !_isTTSEnabled.value
        if (!_isTTSEnabled.value) stopSpeaking()
    }
    
    /**
     * Toggle continuous mode
     */
    fun toggleContinuousMode() {
        _isContinuousModeEnabled.value = !_isContinuousModeEnabled.value
        
        if (_isContinuousModeEnabled.value) {
            startListeningForWakeWord()
        } else {
            stopListening()
        }
    }
    
    /**
     * Set continuous mode
     */
    fun setContinuousMode(enabled: Boolean) {
        _isContinuousModeEnabled.value = enabled
        if (enabled) {
            startListeningForWakeWord()
        } else {
            if (_voiceState.value is VoiceState.WaitingForWakeWord) {
                stopListening()
            }
        }
    }
    
    /**
     * Check if speech recognition is available
     */
    fun isSpeechRecognitionAvailable(): Boolean = isRecognizerAvailable
    
    /**
     * Reset voice state
     */
    fun resetState() {
        _voiceState.value = VoiceState.Idle
        _partialText.value = ""
        isWakeWordMode = false
    }
    
    // Haptic feedback
    private fun vibrateShort() {
        vibrator?.vibrate(VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE))
    }
    
    private fun vibrateConfirm() {
        vibrator?.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 100, 50, 100), -1))
    }
    
    private fun getErrorMessage(error: Int) = when (error) {
        SpeechRecognizer.ERROR_AUDIO -> "Audio error"
        SpeechRecognizer.ERROR_CLIENT -> "Client error"
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "No permission"
        SpeechRecognizer.ERROR_NETWORK -> "Network error"
        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
        SpeechRecognizer.ERROR_NO_MATCH -> "No speech detected"
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy"
        SpeechRecognizer.ERROR_SERVER -> "Server error"
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Speech timeout"
        else -> "Unknown error"
    }
    
    /**
     * Cleanup resources
     */
    fun destroy() {
        speechRecognizer?.destroy()
        speechRecognizer = null
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
    }
}
