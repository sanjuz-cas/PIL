package com.indxai.watch.presentation.theme

import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.wear.compose.material.Colors
import androidx.wear.compose.material.MaterialTheme

/**
 * Indxai Cherry/Beige color palette matching the web app
 */
val Cherry900 = Color(0xFF77243B)
val Cherry800 = Color(0xFF8C2640)
val Cherry700 = Color(0xFFA92B47)
val Cherry600 = Color(0xFFC93556)
val Cherry500 = Color(0xFFE04F6C)
val Cherry400 = Color(0xFFEC7A8F)
val Cherry300 = Color(0xFFF4A9B6)
val Cherry200 = Color(0xFFF9D0D7)
val Cherry100 = Color(0xFFFCE7EA)
val Cherry50 = Color(0xFFFDF2F4)

val Beige50 = Color(0xFFFDFCFB)
val Beige100 = Color(0xFFF9F6F2)
val Beige200 = Color(0xFFEDE8E0)
val Beige300 = Color(0xFFDDD5C9)

val IndxaiWearColors = Colors(
    primary = Cherry600,
    primaryVariant = Cherry800,
    secondary = Cherry400,
    secondaryVariant = Cherry700,
    background = Cherry900,
    surface = Cherry800,
    error = Color(0xFFCF6679),
    onPrimary = Color.White,
    onSecondary = Color.White,
    onBackground = Beige100,
    onSurface = Beige100,
    onError = Color.Black
)

/**
 * Indxai Watch Theme
 */
@Composable
fun IndxaiWatchTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colors = IndxaiWearColors,
        content = content
    )
}
