package com.indxai.pilchat.ui.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.indxai.pilchat.data.ChatMessage
import com.indxai.pilchat.ui.components.MarkdownText
import com.indxai.pilchat.ui.theme.*
import com.indxai.pilchat.voice.VoiceState

/**
 * Main Chat Screen composable
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    onNavigateToSettings: () -> Unit,
    onNavigateToTrain: () -> Unit,
    viewModel: ChatViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val voiceState by viewModel.voiceState.collectAsState()
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    
    // Auto-scroll to bottom when new messages arrive
    LaunchedEffect(uiState.messages.size, uiState.currentResponse) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "PIL-VAE Chat",
                            fontWeight = FontWeight.Bold,
                            color = Beige
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        ConnectionIndicator(status = uiState.connectionStatus)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkBg
                ),
                actions = {
                    // TTS Toggle
                    IconButton(onClick = { viewModel.toggleTTS() }) {
                        Icon(
                            imageVector = if (uiState.isTTSEnabled) 
                                Icons.AutoMirrored.Filled.VolumeUp 
                            else 
                                Icons.AutoMirrored.Filled.VolumeOff,
                            contentDescription = "Toggle TTS",
                            tint = if (uiState.isTTSEnabled) SuccessGreen else TextSecondary
                        )
                    }
                    
                    // Continuous mode toggle
                    IconButton(onClick = { viewModel.toggleContinuousMode() }) {
                        Icon(
                            imageVector = Icons.Default.GraphicEq,
                            contentDescription = "Continuous Mode",
                            tint = if (uiState.isContinuousModeEnabled) CherryRed else TextSecondary
                        )
                    }
                    
                    IconButton(onClick = { viewModel.clearChat() }) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Clear chat",
                            tint = TextSecondary
                        )
                    }
                    IconButton(onClick = onNavigateToTrain) {
                        Icon(
                            imageVector = Icons.Default.School,
                            contentDescription = "Train",
                            tint = CherryRed
                        )
                    }
                    IconButton(onClick = onNavigateToSettings) {
                        Icon(
                            imageVector = Icons.Default.Settings,
                            contentDescription = "Settings",
                            tint = TextSecondary
                        )
                    }
                }
            )
        },
        containerColor = DarkBg
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Voice state indicator
            VoiceStateIndicator(
                voiceState = voiceState,
                partialText = uiState.partialVoiceText
            )
            
            // Messages list
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                contentPadding = PaddingValues(vertical = 16.dp)
            ) {
                // Welcome message if no messages
                if (uiState.messages.isEmpty() && uiState.currentResponse.isEmpty()) {
                    item {
                        WelcomeCard(
                            onSuggestionClick = { suggestion ->
                                viewModel.sendMessage(suggestion)
                            }
                        )
                    }
                }
                
                items(uiState.messages) { message ->
                    ChatBubble(message = message)
                }
                
                // Show streaming response
                if (uiState.currentResponse.isNotEmpty()) {
                    item {
                        ChatBubble(
                            message = ChatMessage(
                                content = uiState.currentResponse,
                                isUser = false
                            ),
                            isStreaming = true
                        )
                    }
                }
                
                // Loading indicator
                if (uiState.isLoading && uiState.currentResponse.isEmpty()) {
                    item {
                        LoadingIndicator()
                    }
                }
            }
            
            // Input area with voice button
            MessageInputWithVoice(
                value = inputText,
                onValueChange = { inputText = it },
                onSend = {
                    if (inputText.isNotBlank()) {
                        viewModel.sendMessage(inputText)
                        inputText = ""
                    }
                },
                onVoiceStart = { viewModel.startVoiceInput() },
                onVoiceStop = { viewModel.stopVoiceInput() },
                voiceState = voiceState,
                isLoading = uiState.isLoading,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            )
        }
    }
}

/**
 * Voice state indicator shown at top when listening/speaking
 */
@Composable
fun VoiceStateIndicator(
    voiceState: VoiceState,
    partialText: String
) {
    AnimatedVisibility(
        visible = voiceState != VoiceState.Idle,
        enter = slideInVertically() + fadeIn(),
        exit = slideOutVertically() + fadeOut()
    ) {
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = when (voiceState) {
                is VoiceState.Listening -> CherryRed.copy(alpha = 0.15f)
                is VoiceState.WaitingForWakeWord -> WarningYellow.copy(alpha = 0.15f)
                is VoiceState.Speaking -> SuccessGreen.copy(alpha = 0.15f)
                is VoiceState.Processing -> Color.Blue.copy(alpha = 0.15f)
                else -> Color.Transparent
            }
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                // Animated indicator
                val infiniteTransition = rememberInfiniteTransition(label = "voice")
                val scale by infiniteTransition.animateFloat(
                    initialValue = 0.8f,
                    targetValue = 1.2f,
                    animationSpec = infiniteRepeatable(
                        animation = tween(500),
                        repeatMode = RepeatMode.Reverse
                    ),
                    label = "pulse"
                )
                
                val (icon, text, color) = when (voiceState) {
                    is VoiceState.Listening -> Triple(Icons.Default.Mic, "Listening...", CherryRed)
                    is VoiceState.WaitingForWakeWord -> Triple(Icons.Default.Hearing, "Say 'Hey PIL'...", WarningYellow)
                    is VoiceState.Speaking -> Triple(Icons.AutoMirrored.Filled.VolumeUp, "Speaking...", SuccessGreen)
                    is VoiceState.Processing -> Triple(Icons.Default.Psychology, "Processing...", Color.Blue)
                    is VoiceState.Error -> Triple(Icons.Default.Error, (voiceState as VoiceState.Error).message, ErrorRed)
                    else -> Triple(Icons.Default.Mic, "", TextSecondary)
                }
                
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = color,
                    modifier = Modifier
                        .size(20.dp)
                        .scale(if (voiceState is VoiceState.Listening) scale else 1f)
                )
                
                Spacer(modifier = Modifier.width(8.dp))
                
                Text(
                    text = if (partialText.isNotEmpty()) partialText else text,
                    color = color,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}

@Composable
fun ConnectionIndicator(status: ConnectionStatus) {
    val color = when (status) {
        ConnectionStatus.CONNECTED -> SuccessGreen
        ConnectionStatus.CONNECTING -> WarningYellow
        ConnectionStatus.OFFLINE -> ErrorRed
    }
    
    val text = when (status) {
        ConnectionStatus.CONNECTED -> "Connected"
        ConnectionStatus.CONNECTING -> "Connecting..."
        ConnectionStatus.OFFLINE -> "Offline"
    }
    
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .background(
                color = color.copy(alpha = 0.2f),
                shape = RoundedCornerShape(12.dp)
            )
            .padding(horizontal = 8.dp, vertical = 4.dp)
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(color)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = text,
            fontSize = 10.sp,
            color = color
        )
    }
}

@Composable
fun WelcomeCard(
    onSuggestionClick: (String) -> Unit = {}
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = DarkSurfaceVariant
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Default.Psychology,
                contentDescription = null,
                tint = CherryRed,
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "Welcome to PIL-VAE Chat",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Beige,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Powered by Pseudoinverse Learning.\nAsk me anything!",
                fontSize = 14.sp,
                color = TextSecondary,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            // Quick suggestions
            QuickSuggestionChips(onSuggestionClick = onSuggestionClick)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun QuickSuggestionChips(
    onSuggestionClick: (String) -> Unit = {}
) {
    val suggestions = listOf(
        "What is PIL learning?",
        "Hello, what can you do?",
        "Explain neural networks"
    )
    
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        suggestions.forEach { suggestion ->
            SuggestionChip(
                onClick = { onSuggestionClick(suggestion) },
                label = { Text(suggestion, fontSize = 12.sp) },
                colors = SuggestionChipDefaults.suggestionChipColors(
                    containerColor = DarkSurface,
                    labelColor = TextSecondary
                )
            )
        }
    }
}

@Composable
fun ChatBubble(
    message: ChatMessage,
    isStreaming: Boolean = false
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (message.isUser) Arrangement.End else Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 320.dp)
                .background(
                    color = if (message.isUser) UserBubble else AIBubble,
                    shape = RoundedCornerShape(
                        topStart = 16.dp,
                        topEnd = 16.dp,
                        bottomStart = if (message.isUser) 16.dp else 4.dp,
                        bottomEnd = if (message.isUser) 4.dp else 16.dp
                    )
                )
                .padding(12.dp)
                .animateContentSize()
        ) {
            Column {
                if (!message.isUser) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(bottom = 4.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.SmartToy,
                            contentDescription = null,
                            tint = CherryRed,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "PIL-VAE",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            color = CherryRed
                        )
                        if (isStreaming) {
                            Spacer(modifier = Modifier.width(8.dp))
                            CircularProgressIndicator(
                                modifier = Modifier.size(12.dp),
                                strokeWidth = 2.dp,
                                color = CherryRed
                            )
                        }
                    }
                }
                
                // Use MarkdownText for AI responses, plain text for user
                if (message.isUser) {
                    Text(
                        text = message.content,
                        color = Beige,
                        fontSize = 14.sp,
                        lineHeight = 20.sp
                    )
                } else {
                    MarkdownText(
                        text = message.content,
                        isUserMessage = false
                    )
                }
            }
        }
    }
}

@Composable
fun LoadingIndicator() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .background(
                    color = AIBubble,
                    shape = RoundedCornerShape(16.dp)
                )
                .padding(16.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    strokeWidth = 2.dp,
                    color = CherryRed
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "Thinking...",
                    color = TextSecondary,
                    fontSize = 14.sp
                )
            }
        }
    }
}

@Composable
fun MessageInputWithVoice(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    onVoiceStart: () -> Unit,
    onVoiceStop: () -> Unit,
    voiceState: VoiceState,
    isLoading: Boolean,
    modifier: Modifier = Modifier
) {
    val isListening = voiceState is VoiceState.Listening || voiceState is VoiceState.WaitingForWakeWord
    
    Row(
        modifier = modifier
            .background(
                color = DarkSurfaceVariant,
                shape = RoundedCornerShape(28.dp)
            )
            .padding(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Voice button
        IconButton(
            onClick = {
                if (isListening) onVoiceStop() else onVoiceStart()
            },
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(
                    if (isListening) CherryRed else DarkSurface
                )
        ) {
            Icon(
                imageVector = if (isListening) Icons.Default.MicOff else Icons.Default.Mic,
                contentDescription = if (isListening) "Stop listening" else "Voice input",
                tint = if (isListening) Beige else TextSecondary
            )
        }
        
        Spacer(modifier = Modifier.width(4.dp))
        
        TextField(
            value = value,
            onValueChange = onValueChange,
            placeholder = {
                Text(
                    "Type or speak...",
                    color = TextDisabled
                )
            },
            colors = TextFieldDefaults.colors(
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent,
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                focusedTextColor = Beige,
                unfocusedTextColor = Beige,
                cursorColor = CherryRed
            ),
            modifier = Modifier.weight(1f),
            maxLines = 4,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { onSend() })
        )
        
        IconButton(
            onClick = onSend,
            enabled = value.isNotBlank() && !isLoading,
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(
                    if (value.isNotBlank() && !isLoading) CherryRed
                    else CherryRed.copy(alpha = 0.3f)
                )
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.Send,
                contentDescription = "Send",
                tint = Beige
            )
        }
    }
}

@Composable
fun MessageInput(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    isLoading: Boolean,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .background(
                color = DarkSurfaceVariant,
                shape = RoundedCornerShape(28.dp)
            )
            .padding(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        TextField(
            value = value,
            onValueChange = onValueChange,
            placeholder = {
                Text(
                    "Type your message...",
                    color = TextDisabled
                )
            },
            colors = TextFieldDefaults.colors(
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent,
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                focusedTextColor = Beige,
                unfocusedTextColor = Beige,
                cursorColor = CherryRed
            ),
            modifier = Modifier.weight(1f),
            maxLines = 4,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { onSend() })
        )
        
        IconButton(
            onClick = onSend,
            enabled = value.isNotBlank() && !isLoading,
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(
                    if (value.isNotBlank() && !isLoading) CherryRed
                    else CherryRed.copy(alpha = 0.3f)
                )
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.Send,
                contentDescription = "Send",
                tint = Beige
            )
        }
    }
}
