package com.indxai.pilchat.ui.screens

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.indxai.pilchat.data.ChatMessage
import com.indxai.pilchat.data.PILApiClient
import com.indxai.pilchat.util.TextCleaner
import com.indxai.pilchat.voice.VoiceCommand
import com.indxai.pilchat.voice.VoiceManager
import com.indxai.pilchat.voice.VoiceState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Connection status for the backend
 */
enum class ConnectionStatus {
    CONNECTING,
    CONNECTED,
    OFFLINE
}

/**
 * UI State for the Chat Screen
 */
data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val isLoading: Boolean = false,
    val connectionStatus: ConnectionStatus = ConnectionStatus.CONNECTING,
    val currentResponse: String = "",
    val errorMessage: String? = null,
    val isTTSEnabled: Boolean = true,
    val isContinuousModeEnabled: Boolean = false,
    val partialVoiceText: String = ""
)

/**
 * ViewModel for the Chat Screen
 */
@HiltViewModel
class ChatViewModel @Inject constructor(
    private val apiClient: PILApiClient,
    val voiceManager: VoiceManager
) : ViewModel() {
    
    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()
    
    // Expose voice state
    val voiceState: StateFlow<VoiceState> = voiceManager.voiceState
    
    init {
        checkConnection()
        observeVoiceState()
    }
    
    private fun observeVoiceState() {
        viewModelScope.launch {
            voiceManager.partialText.collect { partial ->
                _uiState.value = _uiState.value.copy(partialVoiceText = partial)
            }
        }
        
        viewModelScope.launch {
            voiceManager.isTTSEnabled.collect { enabled ->
                _uiState.value = _uiState.value.copy(isTTSEnabled = enabled)
            }
        }
        
        viewModelScope.launch {
            voiceManager.isContinuousModeEnabled.collect { enabled ->
                _uiState.value = _uiState.value.copy(isContinuousModeEnabled = enabled)
            }
        }
    }
    
    /**
     * Check backend connection
     */
    fun checkConnection() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(connectionStatus = ConnectionStatus.CONNECTING)
            
            val result = apiClient.checkHealth()
            _uiState.value = _uiState.value.copy(
                connectionStatus = if (result.isSuccess) ConnectionStatus.CONNECTED else ConnectionStatus.OFFLINE
            )
        }
    }
    
    /**
     * Start voice input (always direct listening, no wake word needed)
     */
    fun startVoiceInput() {
        // Force direct listening mode (no wake word required when user taps mic)
        voiceManager.startListening(
            onQuery = { query ->
                sendMessage(query)
            },
            onCommand = { command ->
                handleVoiceCommand(command)
            }
        )
    }
    
    /**
     * Stop voice input
     */
    fun stopVoiceInput() {
        voiceManager.stopListening()
    }
    
    /**
     * Toggle TTS
     */
    fun toggleTTS() {
        voiceManager.toggleTTS()
    }
    
    /**
     * Toggle continuous mode
     */
    fun toggleContinuousMode() {
        voiceManager.toggleContinuousMode()
    }
    
    /**
     * Handle voice commands
     */
    private fun handleVoiceCommand(command: VoiceCommand) {
        when (command) {
            is VoiceCommand.StopSpeaking -> voiceManager.stopSpeaking()
            is VoiceCommand.ClearChat -> clearChat()
            is VoiceCommand.RepeatLastResponse -> voiceManager.repeatLastResponse()
            is VoiceCommand.EnableContinuousMode -> voiceManager.setContinuousMode(true)
            is VoiceCommand.DisableContinuousMode -> voiceManager.setContinuousMode(false)
            is VoiceCommand.CheckStatus -> {
                val status = if (_uiState.value.connectionStatus == ConnectionStatus.CONNECTED) 
                    "I'm online and ready to help!" 
                else 
                    "I'm currently offline. Please check your connection."
                voiceManager.speak(status)
            }
            else -> {}
        }
    }
    
    /**
     * Send a message with streaming response
     */
    fun sendMessage(query: String) {
        if (query.isBlank()) return
        
        viewModelScope.launch {
            try {
                // Add user message
                val userMessage = ChatMessage(content = query, isUser = true)
                _uiState.value = _uiState.value.copy(
                    messages = _uiState.value.messages + userMessage,
                    isLoading = true,
                    currentResponse = "",
                    errorMessage = null
                )
                
                val responseBuilder = StringBuilder()
                
                // Stream the response
                apiClient.sendChatStream(query).collect { chunk ->
                    responseBuilder.append(chunk)
                    _uiState.value = _uiState.value.copy(
                        currentResponse = TextCleaner.clean(responseBuilder.toString())
                    )
                }
                
                // Clean the final response
                val cleanedResponse = TextCleaner.clean(
                    responseBuilder.toString().ifEmpty { "No response received" }
                )
                
                // Add AI response to messages
                val aiMessage = ChatMessage(
                    content = cleanedResponse,
                    isUser = false
                )
                
                _uiState.value = _uiState.value.copy(
                    messages = _uiState.value.messages + aiMessage,
                    isLoading = false,
                    currentResponse = ""
                )
                
                // Speak the response if TTS is enabled
                if (_uiState.value.isTTSEnabled) {
                    voiceManager.speak(cleanedResponse)
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    errorMessage = "Error: ${e.message ?: "Unknown error"}"
                )
            }
        }
    }
    
    /**
     * Clear chat history
     */
    fun clearChat() {
        _uiState.value = _uiState.value.copy(
            messages = emptyList(),
            errorMessage = null
        )
        voiceManager.speak("Chat cleared")
    }
    
    override fun onCleared() {
        super.onCleared()
        voiceManager.destroy()
    }
}
