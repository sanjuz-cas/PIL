package com.indxai.watch.data.api

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.android.*
import io.ktor.client.plugins.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.logging.*
import io.ktor.client.request.*
import io.ktor.client.request.forms.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.utils.io.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.Json
import javax.inject.Inject
import javax.inject.Singleton

/**
 * API Client for Indxai Backend
 * 
 * Handles communication with the PIL-VAE engine backend.
 * Supports streaming responses via NDJSON.
 */
@Singleton
class IndxaiApiClient @Inject constructor() {
    
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }
    
    private val client = HttpClient(Android) {
        // JSON serialization
        install(ContentNegotiation) {
            json(json)
        }
        
        // Logging for debug
        install(Logging) {
            logger = Logger.ANDROID
            level = LogLevel.INFO
        }
        
        // Timeouts
        install(HttpTimeout) {
            requestTimeoutMillis = 30_000
            connectTimeoutMillis = 10_000
            socketTimeoutMillis = 30_000
        }
        
        // Default request config
        defaultRequest {
            contentType(ContentType.Application.FormUrlEncoded)
        }
    }
    
    companion object {
        // Use 10.0.2.2 for emulator to reach host localhost
        // Change to actual server URL for production
        const val BASE_URL = "http://10.0.2.2:8000"
    }
    
    /**
     * Send chat query and stream response chunks
     */
    fun chat(query: String, mode: String = "assistant"): Flow<String> = flow {
        try {
            val response: HttpResponse = client.submitForm(
                url = "$BASE_URL/v1/chat",
                formParameters = parameters {
                    append("query", query)
                    append("mode", mode)
                }
            )
            
            if (response.status.isSuccess()) {
                // Read streaming NDJSON response
                val channel: ByteReadChannel = response.bodyAsChannel()
                val buffer = StringBuilder()
                
                while (!channel.isClosedForRead) {
                    val byte = channel.readByte()
                    val char = byte.toInt().toChar()
                    
                    if (char == '\n') {
                        val line = buffer.toString().trim()
                        buffer.clear()
                        
                        if (line.isNotEmpty()) {
                            // Parse NDJSON line
                            val chunk = parseChunk(line)
                            if (chunk != null) {
                                emit(chunk)
                            }
                        }
                    } else {
                        buffer.append(char)
                    }
                }
                
                // Emit remaining buffer
                if (buffer.isNotEmpty()) {
                    val chunk = parseChunk(buffer.toString().trim())
                    if (chunk != null) {
                        emit(chunk)
                    }
                }
                
                emit("[DONE]")
            } else {
                emit("ERROR:Server returned ${response.status}")
            }
        } catch (e: Exception) {
            emit("ERROR:${e.message ?: "Connection failed"}")
        }
    }
    
    /**
     * Parse streaming chunk from NDJSON
     */
    private fun parseChunk(line: String): String? {
        return try {
            // Handle raw text chunks (not JSON)
            if (!line.startsWith("{")) {
                return line
            }
            
            // Parse JSON chunk
            val contentRegex = """"content"\s*:\s*"([^"]*)"""".toRegex()
            val match = contentRegex.find(line)
            match?.groupValues?.get(1)
        } catch (e: Exception) {
            line
        }
    }
    
    /**
     * Train new knowledge
     */
    suspend fun trainKnowledge(textData: String): Result<String> {
        return try {
            val response: HttpResponse = client.submitForm(
                url = "$BASE_URL/v1/train-knowledge",
                formParameters = parameters {
                    append("text_data", textData)
                }
            )
            
            if (response.status.isSuccess()) {
                Result.success("Training started")
            } else {
                Result.failure(Exception("Training failed: ${response.status}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Health check - try multiple endpoints
     */
    suspend fun checkHealth(): Boolean {
        return try {
            // Try v1/health first
            val response: HttpResponse = client.get("$BASE_URL/v1/health")
            if (response.status.isSuccess()) {
                android.util.Log.d("IndxaiApiClient", "Health check succeeded")
                true
            } else {
                // Fallback to root endpoint
                val rootResponse: HttpResponse = client.get("$BASE_URL/")
                rootResponse.status.isSuccess()
            }
        } catch (e: Exception) {
            android.util.Log.e("IndxaiApiClient", "Health check failed: ${e.message}")
            false
        }
    }
    
    /**
     * Get engine stats
     */
    suspend fun getStats(): Map<String, Any>? {
        return try {
            val response: HttpResponse = client.get("$BASE_URL/v1/stats")
            if (response.status.isSuccess()) {
                response.body()
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Close client
     */
    fun close() {
        client.close()
    }
}
