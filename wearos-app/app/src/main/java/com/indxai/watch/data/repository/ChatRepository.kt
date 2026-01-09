package com.indxai.watch.data.repository

import com.indxai.watch.data.api.IndxaiApiClient
import com.indxai.watch.data.local.ChatDao
import com.indxai.watch.data.local.ChatMessage
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for Chat Operations
 * 
 * Handles communication between UI and data sources (API + Local DB)
 */
@Singleton
class ChatRepository @Inject constructor(
    private val apiClient: IndxaiApiClient,
    private val chatDao: ChatDao
) {
    
    /**
     * Send chat query and get streaming response
     */
    fun chat(query: String, mode: String = "assistant"): Flow<String> {
        return apiClient.chat(query, mode)
    }
    
    /**
     * Train new knowledge
     */
    suspend fun trainKnowledge(textData: String): Result<String> {
        return apiClient.trainKnowledge(textData)
    }
    
    /**
     * Check backend health
     */
    suspend fun checkHealth(): Boolean {
        return apiClient.checkHealth()
    }
    
    /**
     * Get recent messages from local DB
     */
    fun getMessages(limit: Int = 50): Flow<List<ChatMessage>> {
        return chatDao.getRecentMessages(limit)
    }
    
    /**
     * Save message to local DB
     */
    suspend fun saveMessage(content: String, isUser: Boolean, sessionId: String? = null) {
        val message = ChatMessage(
            content = content,
            isUser = isUser,
            sessionId = sessionId
        )
        chatDao.insertMessage(message)
    }
    
    /**
     * Clear all messages
     */
    suspend fun clearMessages() {
        chatDao.clearAllMessages()
    }
    
    /**
     * Get message count
     */
    suspend fun getMessageCount(): Int {
        return chatDao.getMessageCount()
    }
}
