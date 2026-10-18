package app.folio.ui

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// Палитра тёмной темы (обычной)
private val DarkScheme = darkColorScheme(
    primary = Color(0xFF9FA8DA),
    secondary = Color(0xFFFFCC80)
)

// AMOLED: чисто чёрный фон, экономим батарею на OLED
private val AmoledScheme = darkColorScheme(
    primary = Color(0xFF9FA8DA),
    secondary = Color(0xFFFFCC80),
    background = Color(0xFF000000),
    surface = Color(0xFF000000),
    surfaceVariant = Color(0xFF0D0D0D),
    surfaceContainer = Color(0xFF0D0D0D),
    surfaceContainerLow = Color(0xFF000000),
    surfaceContainerHigh = Color(0xFF1A1A1A),
    surfaceContainerHighest = Color(0xFF222222),
)

// Бумага: тёплый светло-бежевый фон, не слепит
private val PaperScheme = lightColorScheme(
    primary = Color(0xFF5D4037),
    secondary = Color(0xFF795548),
    background = Color(0xFFF5ECD7),
    surface = Color(0xFFF5ECD7),
    surfaceVariant = Color(0xFFEDE0C4),
    surfaceContainer = Color(0xFFEDE0C4),
    surfaceContainerLow = Color(0xFFF5ECD7),
    surfaceContainerHigh = Color(0xFFE0CFB0),
    surfaceContainerHighest = Color(0xFFD7C19A),
    onBackground = Color(0xFF2C1B0E),
    onSurface = Color(0xFF2C1B0E),
    onSurfaceVariant = Color(0xFF5D4037),
)

// Светлая стандартная
private val LightScheme = lightColorScheme(
    primary = Color(0xFF3F51B5),
    secondary = Color(0xFFEF6C00)
)

@Composable
fun FolioTheme(content: @Composable () -> Unit) {
    val dark = Settings.isDark(isSystemInDarkTheme())
    val ctx = LocalContext.current
    val scheme = when {
        Settings.theme == "amoled" -> AmoledScheme
        Settings.theme == "paper" -> PaperScheme
        Build.VERSION.SDK_INT >= 31 && Settings.dynamicColors ->
            if (dark) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        dark -> DarkScheme
        else -> LightScheme
    }
    MaterialTheme(colorScheme = scheme, content = content)
}
