package com.indxai.pilchat.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.indxai.pilchat.ui.theme.*

/**
 * Training Screen for teaching new knowledge to PIL-VAE
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainScreen(
    onNavigateBack: () -> Unit,
    viewModel: TrainViewModel = hiltViewModel()
) {
    val isLoading by viewModel.isLoading.collectAsState()
    val result by viewModel.result.collectAsState()
    val isSuccess by viewModel.isSuccess.collectAsState()
    val trainingStatus by viewModel.trainingStatus.collectAsState()
    
    var trainingText by remember { mutableStateOf("") }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Train Knowledge",
                        fontWeight = FontWeight.Bold,
                        color = Beige
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = Beige
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { viewModel.fetchTrainingStatus() }) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Refresh Stats",
                            tint = Beige
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkBg
                )
            )
        },
        containerColor = DarkBg
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Knowledge Status Card
            trainingStatus?.let { status ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = CherryRed.copy(alpha = 0.15f)
                    ),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = Icons.Default.Psychology,
                                contentDescription = null,
                                tint = CherryRed,
                                modifier = Modifier.size(28.dp)
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Text(
                                text = "Knowledge Base Status",
                                color = Beige,
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.weight(1f))
                            if (status.vaeTrained) {
                                Icon(
                                    imageVector = Icons.Default.CheckCircle,
                                    contentDescription = "Trained",
                                    tint = SuccessGreen,
                                    modifier = Modifier.size(20.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(12.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceEvenly
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(
                                    text = "${status.totalKnowledge}",
                                    color = CherryRed,
                                    fontSize = 24.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "Total Entries",
                                    color = TextSecondary,
                                    fontSize = 11.sp
                                )
                            }
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(
                                    text = "${status.userTrainedCount}",
                                    color = SuccessGreen,
                                    fontSize = 24.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "User Trained",
                                    color = TextSecondary,
                                    fontSize = 11.sp
                                )
                            }
                        }
                    }
                }
            }
            
            // Info Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurfaceVariant
                ),
                shape = RoundedCornerShape(16.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.School,
                        contentDescription = null,
                        tint = CherryRed,
                        modifier = Modifier.size(32.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(
                            text = "Teach PIL-VAE",
                            color = Beige,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Enter text to train the AI with new knowledge",
                            color = TextSecondary,
                            fontSize = 12.sp
                        )
                    }
                }
            }
            
            // Training Input
            OutlinedTextField(
                value = trainingText,
                onValueChange = { trainingText = it },
                label = { Text("Training Text") },
                placeholder = { Text("Enter knowledge to teach...") },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = CherryRed,
                    unfocusedBorderColor = TextDisabled,
                    focusedLabelColor = CherryRed,
                    unfocusedLabelColor = TextSecondary,
                    focusedTextColor = Beige,
                    unfocusedTextColor = Beige,
                    cursorColor = CherryRed,
                    focusedPlaceholderColor = TextDisabled,
                    unfocusedPlaceholderColor = TextDisabled
                ),
                shape = RoundedCornerShape(12.dp)
            )
            
            // Character count
            Text(
                text = "${trainingText.length} characters",
                color = TextSecondary,
                fontSize = 12.sp,
                modifier = Modifier.align(Alignment.End)
            )
            
            // Train Button
            Button(
                onClick = {
                    if (trainingText.isNotBlank()) {
                        viewModel.trainKnowledge(trainingText)
                    }
                },
                enabled = trainingText.isNotBlank() && !isLoading,
                colors = ButtonDefaults.buttonColors(
                    containerColor = CherryRed,
                    disabledContainerColor = CherryRed.copy(alpha = 0.3f)
                ),
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp)
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = Beige,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Training...")
                } else {
                    Icon(
                        imageVector = Icons.Default.Upload,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Start Training")
                }
            }
            
            // Result Card
            result?.let { resultText ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isSuccess == true) 
                            SuccessGreen.copy(alpha = 0.2f) 
                        else 
                            ErrorRed.copy(alpha = 0.2f)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = if (isSuccess == true) 
                                Icons.Default.CheckCircle 
                            else 
                                Icons.Default.Error,
                            contentDescription = null,
                            tint = if (isSuccess == true) SuccessGreen else ErrorRed,
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            text = resultText,
                            color = Beige,
                            fontSize = 14.sp
                        )
                    }
                }
                
                // Clear and new training button
                if (isSuccess == true) {
                    OutlinedButton(
                        onClick = {
                            trainingText = ""
                            viewModel.clearResult()
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = CherryRed
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("Train More Knowledge")
                    }
                }
            }
            
            // Tips
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface
                ),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Text(
                        text = "💡 Tips for better training:",
                        color = Beige,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "• Use clear, factual statements\n" +
                               "• Include context and examples\n" +
                               "• Longer texts provide more knowledge\n" +
                               "• Training happens in the background",
                        color = TextSecondary,
                        fontSize = 12.sp,
                        lineHeight = 18.sp
                    )
                }
            }
        }
    }
}
