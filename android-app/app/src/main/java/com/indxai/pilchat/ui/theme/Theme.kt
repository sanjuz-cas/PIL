package com.indxai.pilchat.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val PILDarkColorScheme = darkColorScheme(
    primary = CherryRed,
    onPrimary = Beige,
    primaryContainer = CherryRedDark,
    onPrimaryContainer = Beige,
    
    secondary = DarkSurfaceVariant,
    onSecondary = TextPrimary,
    secondaryContainer = DarkSurface,
    onSecondaryContainer = TextSecondary,
    
    tertiary = CherryRedLight,
    onTertiary = DarkBg,
    
    background = DarkBg,
    onBackground = TextPrimary,
    
    surface = DarkSurface,
    onSurface = TextPrimary,
    surfaceVariant = DarkSurfaceVariant,
    onSurfaceVariant = TextSecondary,
    
    error = ErrorRed,
    onError = Beige,
    
    outline = TextDisabled,
    outlineVariant = DarkSurfaceVariant
)

@Composable
fun PILChatTheme(
    @Suppress("UNUSED_PARAMETER")
    darkTheme: Boolean = true, // Always dark theme for PIL
    content: @Composable () -> Unit
) {
    val colorScheme = PILDarkColorScheme
    val view = LocalView.current
    
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            window.navigationBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = false
            WindowCompat.getInsetsController(window, view).isAppearanceLightNavigationBars = false
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
