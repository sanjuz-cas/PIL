package com.indxai.pilchat.data

import com.indxai.pilchat.BuildConfig
import com.indxai.pilchat.util.TextCleaner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import okhttp3.FormBody
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.BufferedReader
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Data class for health check response
 */
data class HealthResponse(
    val status: String,
    val engine: String,
    val version: String? = null
)

/**
 * Data class for training status
 */
data class TrainingStatus(
    val totalKnowledge: Int,
    val userTrainedCount: Int,
    val vaeTrained: Boolean,
    val sourceBreakdown: Map<String, Int> = emptyMap()
)

/**
 * Data class for chat messages
 */
data class ChatMessage(
    val content: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * API Client for PIL-VAE Backend
 * Handles all communication with the Python FastAPI server
 */
@Singleton
class PILApiClient @Inject constructor() {
    
    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    
    private val baseUrl: String
        get() = BuildConfig.API_BASE_URL
    
    /**
     * Check if the backend server is healthy
     */
    suspend fun checkHealth(): Result<HealthResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/v1/health")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            
            if (response.isSuccessful) {
                val body = response.body?.string() ?: ""
                // Simple JSON parsing
                val status = extractJsonValue(body, "status") ?: "unknown"
                val engine = extractJsonValue(body, "engine") ?: "unknown"
                val version = extractJsonValue(body, "version")
                
                Result.success(HealthResponse(status, engine, version))
            } else {
                Result.failure(Exception("Server returned ${response.code}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Send a chat message and stream the response
     */
    fun sendChatStream(query: String, mode: String = "assistant"): Flow<String> = flow {
        val formBody = FormBody.Builder()
            .add("query", query)
            .add("mode", mode)
            .build()
        
        val request = Request.Builder()
            .url("$baseUrl/v1/chat")
            .post(formBody)
            .build()
        
        try {
            val response = client.newCall(request).execute()
            
            if (response.isSuccessful) {
                response.body?.let { body ->
                    val reader = BufferedReader(body.charStream())
                    var line: String?
                    
                    while (reader.readLine().also { line = it } != null) {
                        line?.let { jsonLine ->
                            if (jsonLine.isNotBlank() && !jsonLine.contains("[DONE]")) {
                                val content = extractJsonValue(jsonLine, "content")
                                if (!content.isNullOrEmpty()) {
                                    emit(content.replace("\\n", "\n"))
                                }
                            }
                        }
                    }
                }
            } else {
                emit("Error: Server returned ${response.code}")
            }
        } catch (e: Exception) {
            emit("Error: ${e.message}\n\nMake sure backend is running on $baseUrl")
        }
    }.flowOn(Dispatchers.IO)
    
    /**
     * Send a non-streaming chat message
     */
    suspend fun sendChat(query: String, mode: String = "assistant"): Result<String> = withContext(Dispatchers.IO) {
        try {
            val formBody = FormBody.Builder()
                .add("query", query)
                .add("mode", mode)
                .build()
            
            val request = Request.Builder()
                .url("$baseUrl/v1/chat")
                .post(formBody)
                .build()
            
            val response = client.newCall(request).execute()
            
            if (response.isSuccessful) {
                val body = response.body?.string() ?: ""
                val fullResponse = StringBuilder()
                
                // Parse NDJSON response
                body.lines().forEach { line ->
                    if (line.isNotBlank() && !line.contains("[DONE]")) {
                        val content = extractJsonValue(line, "content")
                        if (!content.isNullOrEmpty()) {
                            fullResponse.append(content.replace("\\n", "\n"))
                        }
                    }
                }
                
                Result.success(TextCleaner.clean(fullResponse.toString().ifEmpty { "No response received" }))
            } else {
                Result.failure(Exception("Server returned ${response.code}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Train the AI with new knowledge
     */
    suspend fun trainKnowledge(textData: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            val formBody = FormBody.Builder()
                .add("text_data", textData)
                .build()
            
            val request = Request.Builder()
                .url("$baseUrl/v1/train-knowledge")
                .post(formBody)
                .build()
            
            val response = client.newCall(request).execute()
            
            if (response.isSuccessful) {
                val body = response.body?.string() ?: ""
                val message = extractJsonValue(body, "message") ?: "Training started"
                Result.success(message)
            } else {
                Result.failure(Exception("Server returned ${response.code}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Get training status with knowledge count breakdown
     */
    suspend fun getTrainingStatus(): Result<TrainingStatus> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/v1/training-status")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            
            if (response.isSuccessful) {
                val body = response.body?.string() ?: ""
                val totalKnowledge = extractJsonInt(body, "total_knowledge") ?: 0
                val userTrained = extractJsonInt(body, "user_trained_count") ?: 0
                val vaeTrained = body.contains("\"vae_trained\": true") || 
                                 body.contains("\"vae_trained\":true")
                
                Result.success(TrainingStatus(
                    totalKnowledge = totalKnowledge,
                    userTrainedCount = userTrained,
                    vaeTrained = vaeTrained
                ))
            } else {
                Result.failure(Exception("Server returned ${response.code}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Simple JSON value extractor
     */
    private fun extractJsonValue(json: String, key: String): String? {
        val regex = """"$key"\s*:\s*"([^"]*)"""".toRegex()
        return regex.find(json)?.groupValues?.get(1)
    }
    
    /**
     * Extract integer value from JSON
     */
    private fun extractJsonInt(json: String, key: String): Int? {
        val regex = """"$key"\s*:\s*(\d+)""".toRegex()
        return regex.find(json)?.groupValues?.get(1)?.toIntOrNull()
    }
}
