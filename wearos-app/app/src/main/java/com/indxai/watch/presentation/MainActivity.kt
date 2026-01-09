package com.indxai.watch.presentation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.wear.compose.material.*
import com.indxai.watch.BuildConfig
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Simple PIL-VAE Chat Activity for WearOS
 * Connects directly to the Python backend
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SimpleChatScreen()
        }
    }
}

@Composable
fun SimpleChatScreen() {
    val scope = rememberCoroutineScope()
    var response by remember { mutableStateOf("Type a message or tap a button") }
    var isLoading by remember { mutableStateOf(false) }
    var isConnected by remember { mutableStateOf(false) }
    var userInput by remember { mutableStateOf("") }
    val scrollState = rememberScrollState()
    
    // Check connection on start
    LaunchedEffect(Unit) {
        isConnected = checkHealth()
    }
    
    val darkBg = Color(0xFF1A1A2E)
    val cherry = Color(0xFFE63946)
    val beige = Color(0xFFF1FAEE)
    
    Scaffold(
        timeText = { TimeText() }
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(darkBg)
                .padding(8.dp)
                .verticalScroll(scrollState),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(24.dp))
            
            // Title
            Text(
                text = "PIL-VAE",
                color = beige,
                fontSize = 18.sp,
                textAlign = TextAlign.Center
            )
            
            // Connection status
            Text(
                text = if (isConnected) "● Connected" else "○ Offline",
                color = if (isConnected) Color.Green else Color.Gray,
                fontSize = 10.sp
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Text input field
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp)
                    .background(Color(0xFF2A2A4E), RoundedCornerShape(12.dp))
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                if (userInput.isEmpty()) {
                    Text(
                        text = "Type your question...",
                        color = beige.copy(alpha = 0.5f),
                        fontSize = 12.sp
                    )
                }
                BasicTextField(
                    value = userInput,
                    onValueChange = { userInput = it },
                    textStyle = TextStyle(
                        color = beige,
                        fontSize = 12.sp
                    ),
                    cursorBrush = SolidColor(cherry),
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = false,
                    maxLines = 3
                )
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Send button
            CompactChip(
                onClick = {
                    if (userInput.isNotBlank()) {
                        scope.launch {
                            isLoading = true
                            response = sendChat(userInput)
                            userInput = ""
                            isLoading = false
                        }
                    }
                },
                label = { Text("Send", fontSize = 11.sp) },
                colors = ChipDefaults.primaryChipColors(backgroundColor = cherry),
                enabled = !isLoading && userInput.isNotBlank()
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Response area
            Text(
                text = if (isLoading) "Thinking..." else response,
                color = beige.copy(alpha = 0.9f),
                fontSize = 11.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 4.dp)
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Quick buttons row
            Text(
                text = "Quick Actions:",
                color = beige.copy(alpha = 0.6f),
                fontSize = 9.sp
            )
            
            Spacer(modifier = Modifier.height(4.dp))
            
            // Quick chat buttons - more compact
            CompactChip(
                onClick = {
                    scope.launch {
                        isLoading = true
                        response = sendChat("Hello, what can you do?")
                        isLoading = false
                    }
                },
                label = { Text("Hello", fontSize = 10.sp) },
                colors = ChipDefaults.secondaryChipColors(),
                enabled = !isLoading
            )
            
            Spacer(modifier = Modifier.height(4.dp))
            
            CompactChip(
                onClick = {
                    scope.launch {
                        isLoading = true
                        response = sendChat("What is PIL learning?")
                        isLoading = false
                    }
                },
                label = { Text("PIL Info", fontSize = 10.sp) },
                colors = ChipDefaults.secondaryChipColors(),
                enabled = !isLoading
            )
            
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

/**
 * Send chat message to PIL-VAE backend
 */
suspend fun sendChat(query: String): String = withContext(Dispatchers.IO) {
    try {
        val baseUrl = BuildConfig.API_BASE_URL
        val url = URL("$baseUrl/v1/chat")
        val connection = url.openConnection() as HttpURLConnection
        
        connection.requestMethod = "POST"
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
        connection.connectTimeout = 10000
        connection.readTimeout = 30000
        
        // Send form data
        val postData = "query=${URLEncoder.encode(query, "UTF-8")}&mode=assistant"
        OutputStreamWriter(connection.outputStream).use { writer ->
            writer.write(postData)
            writer.flush()
        }
        
        // Read streaming response
        if (connection.responseCode == 200) {
            val responseText = StringBuilder()
            BufferedReader(connection.inputStream.reader()).use { reader ->
                reader.forEachLine { line ->
                    // Parse NDJSON - each line is {"type":"token","content":"..."}
                    if (line.isNotBlank() && !line.contains("[DONE]")) {
                        // Extract content from JSON token
                        val content = extractContent(line)
                        if (content.isNotEmpty()) {
                            responseText.append(content)
                        }
                    }
                }
            }
            responseText.toString().ifEmpty { "No response received" }
        } else {
            "Error: ${connection.responseCode}"
        }
    } catch (e: Exception) {
        "Error: ${e.message}\n\nMake sure backend is running on ${BuildConfig.API_BASE_URL}"
    }
}

/**
 * Extract content from streaming JSON token
 * Parses: {"type":"token","content":"Hello"}
 */
fun extractContent(json: String): String {
    return try {
        // Simple regex to extract content value
        val regex = """"content"\s*:\s*"([^"]*)"""".toRegex()
        val match = regex.find(json)
        match?.groupValues?.get(1)?.replace("\\n", "\n") ?: ""
    } catch (e: Exception) {
        ""
    }
}

/**
 * Check backend health
 */
suspend fun checkHealth(): Boolean = withContext(Dispatchers.IO) {
    try {
        val baseUrl = BuildConfig.API_BASE_URL
        val url = URL("$baseUrl/v1/health")
        val connection = url.openConnection() as HttpURLConnection
        connection.connectTimeout = 5000
        connection.readTimeout = 5000
        connection.responseCode == 200
    } catch (e: Exception) {
        false
    }
}
