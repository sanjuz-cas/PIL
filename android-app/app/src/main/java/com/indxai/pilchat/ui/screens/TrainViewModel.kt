package com.indxai.pilchat.ui.screens

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.indxai.pilchat.data.PILApiClient
import com.indxai.pilchat.data.TrainingStatus
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * ViewModel for Training Screen
 */
@HiltViewModel
class TrainViewModel @Inject constructor(
    private val apiClient: PILApiClient
) : ViewModel() {
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _result = MutableStateFlow<String?>(null)
    val result: StateFlow<String?> = _result.asStateFlow()
    
    private val _isSuccess = MutableStateFlow<Boolean?>(null)
    val isSuccess: StateFlow<Boolean?> = _isSuccess.asStateFlow()
    
    private val _trainingStatus = MutableStateFlow<TrainingStatus?>(null)
    val trainingStatus: StateFlow<TrainingStatus?> = _trainingStatus.asStateFlow()
    
    init {
        fetchTrainingStatus()
    }
    
    fun fetchTrainingStatus() {
        viewModelScope.launch {
            val response = apiClient.getTrainingStatus()
            _trainingStatus.value = response.getOrNull()
        }
    }
    
    fun trainKnowledge(text: String) {
        viewModelScope.launch {
            _isLoading.value = true
            _result.value = null
            
            val response = apiClient.trainKnowledge(text)
            
            _isLoading.value = false
            _isSuccess.value = response.isSuccess
            _result.value = response.getOrElse { it.message ?: "Training failed" }
            
            // Refresh status after training
            if (response.isSuccess) {
                // Wait a bit for backend to process
                kotlinx.coroutines.delay(1500)
                fetchTrainingStatus()
            }
        }
    }
    
    fun clearResult() {
        _result.value = null
        _isSuccess.value = null
    }
}
