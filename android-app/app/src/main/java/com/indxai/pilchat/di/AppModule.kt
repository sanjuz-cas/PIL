package com.indxai.pilchat.di

import com.indxai.pilchat.data.PILApiClient
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt Module for providing app-wide dependencies
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    
    @Provides
    @Singleton
    fun providePILApiClient(): PILApiClient {
        return PILApiClient()
    }
}
