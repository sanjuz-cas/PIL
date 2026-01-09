package com.indxai.watch.data.repository

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import java.io.IOException
import javax.inject.Inject
import javax.inject.Singleton

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

/**
 * Repository for App Settings
 * 
 * Uses DataStore for persistent settings storage
 */
@Singleton
class SettingsRepository @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    companion object {
        private val TTS_ENABLED = booleanPreferencesKey("tts_enabled")
        private val VOICE_SPEED = floatPreferencesKey("voice_speed")
        private val VOICE_PITCH = floatPreferencesKey("voice_pitch")
        private val SERVER_URL = stringPreferencesKey("server_url")
        private val AUTO_LISTEN = booleanPreferencesKey("auto_listen")
        private val HAPTIC_FEEDBACK = booleanPreferencesKey("haptic_feedback")
        
        const val DEFAULT_SERVER_URL = "http://10.0.2.2:8000"
    }
    
    // ========== TTS Enabled ==========
    
    fun getTtsEnabled(): Flow<Boolean> {
        return context.dataStore.data
            .catch { exception ->
                if (exception is IOException) {
                    emit(emptyPreferences())
                } else {
                    throw exception
                }
            }
            .map { preferences ->
                preferences[TTS_ENABLED] ?: true
            }
    }
    
    suspend fun setTtsEnabled(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[TTS_ENABLED] = enabled
        }
    }
    
    // ========== Voice Speed ==========
    
    fun getVoiceSpeed(): Flow<Float> {
        return context.dataStore.data
            .catch { exception ->
                if (exception is IOException) {
                    emit(emptyPreferences())
                } else {
                    throw exception
                }
            }
            .map { preferences ->
                preferences[VOICE_SPEED] ?: 1.0f
            }
    }
    
    suspend fun setVoiceSpeed(speed: Float) {
        context.dataStore.edit { preferences ->
            preferences[VOICE_SPEED] = speed.coerceIn(0.5f, 2.0f)
        }
    }
    
    // ========== Voice Pitch ==========
    
    fun getVoicePitch(): Flow<Float> {
        return context.dataStore.data
            .catch { exception ->
                if (exception is IOException) {
                    emit(emptyPreferences())
                } else {
                    throw exception
                }
            }
            .map { preferences ->
                preferences[VOICE_PITCH] ?: 1.0f
            }
    }
    
    suspend fun setVoicePitch(pitch: Float) {
        context.dataStore.edit { preferences ->
            preferences[VOICE_PITCH] = pitch.coerceIn(0.5f, 2.0f)
        }
    }
    
    // ========== Server URL ==========
    
    fun getServerUrl(): Flow<String> {
        return context.dataStore.data
            .catch { exception ->
                if (exception is IOException) {
                    emit(emptyPreferences())
                } else {
                    throw exception
                }
            }
            .map { preferences ->
                preferences[SERVER_URL] ?: DEFAULT_SERVER_URL
            }
    }
    
    suspend fun setServerUrl(url: String) {
        context.dataStore.edit { preferences ->
            preferences[SERVER_URL] = url
        }
    }
    
    // ========== Auto Listen ==========
    
    fun getAutoListen(): Flow<Boolean> {
        return context.dataStore.data
            .catch { exception ->
                if (exception is IOException) {
                    emit(emptyPreferences())
                } else {
                    throw exception
                }
            }
            .map { preferences ->
                preferences[AUTO_LISTEN] ?: false
            }
    }
    
    suspend fun setAutoListen(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[AUTO_LISTEN] = enabled
        }
    }
    
    // ========== Haptic Feedback ==========
    
    fun getHapticFeedback(): Flow<Boolean> {
        return context.dataStore.data
            .catch { exception ->
                if (exception is IOException) {
                    emit(emptyPreferences())
                } else {
                    throw exception
                }
            }
            .map { preferences ->
                preferences[HAPTIC_FEEDBACK] ?: true
            }
    }
    
    suspend fun setHapticFeedback(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[HAPTIC_FEEDBACK] = enabled
        }
    }
}
