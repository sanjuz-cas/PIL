package com.indxai.watch.di

import android.content.Context
import androidx.room.Room
import com.indxai.watch.data.api.IndxaiApiClient
import com.indxai.watch.data.local.ChatDao
import com.indxai.watch.data.local.ChatDatabase
import com.indxai.watch.data.repository.ChatRepository
import com.indxai.watch.data.repository.SettingsRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt Dependency Injection Module
 * 
 * Provides singleton instances for app-wide dependencies.
 * Voice features disabled for emulator compatibility.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    
    // ========== Database ==========
    
    @Provides
    @Singleton
    fun provideChatDatabase(
        @ApplicationContext context: Context
    ): ChatDatabase {
        return Room.databaseBuilder(
            context,
            ChatDatabase::class.java,
            "indxai_chat.db"
        )
            .fallbackToDestructiveMigration()
            .build()
    }
    
    @Provides
    @Singleton
    fun provideChatDao(database: ChatDatabase): ChatDao {
        return database.chatDao()
    }
    
    // ========== API Client ==========
    
    @Provides
    @Singleton
    fun provideIndxaiApiClient(): IndxaiApiClient {
        return IndxaiApiClient()
    }
    
    // ========== Repositories ==========
    
    @Provides
    @Singleton
    fun provideChatRepository(
        apiClient: IndxaiApiClient,
        chatDao: ChatDao
    ): ChatRepository {
        return ChatRepository(apiClient, chatDao)
    }
    
    @Provides
    @Singleton
    fun provideSettingsRepository(
        @ApplicationContext context: Context
    ): SettingsRepository {
        return SettingsRepository(context)
    }
}
