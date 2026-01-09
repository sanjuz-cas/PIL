package com.indxai.pilchat.ui

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.indxai.pilchat.ui.screens.ChatScreen
import com.indxai.pilchat.ui.screens.SettingsScreen
import com.indxai.pilchat.ui.screens.TrainScreen

/**
 * Navigation routes for the PIL Chat app
 */
sealed class Screen(val route: String) {
    object Chat : Screen("chat")
    object Settings : Screen("settings")
    object Train : Screen("train")
}

@Composable
fun PILChatNavigation() {
    val navController = rememberNavController()
    
    NavHost(
        navController = navController,
        startDestination = Screen.Chat.route
    ) {
        composable(Screen.Chat.route) {
            ChatScreen(
                onNavigateToSettings = { navController.navigate(Screen.Settings.route) },
                onNavigateToTrain = { navController.navigate(Screen.Train.route) }
            )
        }
        
        composable(Screen.Settings.route) {
            SettingsScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        composable(Screen.Train.route) {
            TrainScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
    }
}
