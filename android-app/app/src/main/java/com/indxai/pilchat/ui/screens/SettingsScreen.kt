package com.indxai.pilchat.ui.screens

import androidx.compose.foundation.background
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
import com.indxai.pilchat.BuildConfig
import com.indxai.pilchat.ui.theme.*

/**
 * Settings Screen for the PIL Chat app
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit,
    viewModel: ChatViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Settings",
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
            // Connection Status Card
            SettingsCard(title = "Connection") {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "Backend Status",
                            color = Beige,
                            fontSize = 14.sp
                        )
                        Text(
                            text = BuildConfig.API_BASE_URL,
                            color = TextSecondary,
                            fontSize = 12.sp
                        )
                    }
                    
                    val statusColor = when (uiState.connectionStatus) {
                        ConnectionStatus.CONNECTED -> SuccessGreen
                        ConnectionStatus.CONNECTING -> WarningYellow
                        ConnectionStatus.OFFLINE -> ErrorRed
                    }
                    
                    Box(
                        modifier = Modifier
                            .background(
                                color = statusColor.copy(alpha = 0.2f),
                                shape = RoundedCornerShape(8.dp)
                            )
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = when (uiState.connectionStatus) {
                                ConnectionStatus.CONNECTED -> "Connected"
                                ConnectionStatus.CONNECTING -> "Connecting"
                                ConnectionStatus.OFFLINE -> "Offline"
                            },
                            color = statusColor,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(12.dp))
                
                Button(
                    onClick = { viewModel.checkConnection() },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = CherryRed
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Reconnect")
                }
            }
            
            // About Card
            SettingsCard(title = "About") {
                SettingsRow(
                    icon = Icons.Default.Info,
                    title = "App Version",
                    subtitle = BuildConfig.VERSION_NAME
                )
                Divider(color = DarkSurface, modifier = Modifier.padding(vertical = 8.dp))
                SettingsRow(
                    icon = Icons.Default.Memory,
                    title = "Engine",
                    subtitle = "PIL-VAE Hybrid"
                )
                Divider(color = DarkSurface, modifier = Modifier.padding(vertical = 8.dp))
                SettingsRow(
                    icon = Icons.Default.Code,
                    title = "Architecture",
                    subtitle = "Pseudoinverse Learning"
                )
            }
            
            // Help Card
            SettingsCard(title = "Help") {
                Text(
                    text = "How to Connect:",
                    color = Beige,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "1. Start the backend server on your PC:\n" +
                           "   uvicorn app.main:app --host 0.0.0.0 --port 8000\n\n" +
                           "2. Find your PC's IP address:\n" +
                           "   Run 'ipconfig' in terminal\n\n" +
                           "3. Update API_BASE_URL in build.gradle.kts\n" +
                           "   with your PC's IP address\n\n" +
                           "4. Ensure phone and PC are on same WiFi",
                    color = TextSecondary,
                    fontSize = 12.sp,
                    lineHeight = 18.sp
                )
            }
        }
    }
}

@Composable
fun SettingsCard(
    title: String,
    content: @Composable ColumnScope.() -> Unit
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
                .padding(16.dp)
        ) {
            Text(
                text = title,
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium,
                color = CherryRed,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            content()
        }
    }
}

@Composable
fun SettingsRow(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = TextSecondary,
            modifier = Modifier.size(20.dp)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                color = Beige,
                fontSize = 14.sp
            )
            Text(
                text = subtitle,
                color = TextSecondary,
                fontSize = 12.sp
            )
        }
    }
}
