package com.indxai.pilchat.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.ClickableText
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.*
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indxai.pilchat.util.TextCleaner
import kotlinx.coroutines.launch

/**
 * Code block theme colors
 */
object CodeTheme {
    // Dark theme (Dracula-inspired)
    val Background = Color(0xFF282A36)
    val HeaderBackground = Color(0xFF21222C)
    val LineNumber = Color(0xFF6272A4)
    val Text = Color(0xFFF8F8F2)
    
    // Syntax highlighting
    val Keyword = Color(0xFFFF79C6)      // Pink - if, for, class, def, val, var
    val String = Color(0xFFF1FA8C)        // Yellow - "strings"
    val Comment = Color(0xFF6272A4)       // Gray - // comments
    val Function = Color(0xFF50FA7B)      // Green - function names
    val Number = Color(0xFFBD93F9)        // Purple - numbers
    val Type = Color(0xFF8BE9FD)          // Cyan - types, classes
    val Operator = Color(0xFFFF79C6)      // Pink - operators
    val Variable = Color(0xFFFFB86C)      // Orange - variables
    val Annotation = Color(0xFF50FA7B)    // Green - @annotations
}

/**
 * Language-specific keywords for syntax highlighting
 */
object SyntaxPatterns {
    val KOTLIN_KEYWORDS = setOf(
        "fun", "val", "var", "if", "else", "when", "for", "while", "do",
        "return", "break", "continue", "class", "interface", "object",
        "sealed", "data", "enum", "annotation", "companion", "private",
        "protected", "public", "internal", "override", "open", "final",
        "abstract", "suspend", "inline", "crossinline", "noinline",
        "reified", "import", "package", "true", "false", "null", "this",
        "super", "is", "in", "as", "by", "get", "set", "init", "try",
        "catch", "finally", "throw", "typealias", "lateinit", "lazy"
    )
    
    val PYTHON_KEYWORDS = setOf(
        "def", "class", "if", "elif", "else", "for", "while", "try",
        "except", "finally", "with", "as", "import", "from", "return",
        "yield", "raise", "break", "continue", "pass", "lambda", "and",
        "or", "not", "in", "is", "True", "False", "None", "self", "async",
        "await", "global", "nonlocal", "assert", "del"
    )
    
    val JAVASCRIPT_KEYWORDS = setOf(
        "function", "const", "let", "var", "if", "else", "for", "while",
        "do", "switch", "case", "default", "break", "continue", "return",
        "try", "catch", "finally", "throw", "class", "extends", "new",
        "this", "super", "import", "export", "from", "async", "await",
        "true", "false", "null", "undefined", "typeof", "instanceof"
    )
    
    val JAVA_KEYWORDS = setOf(
        "public", "private", "protected", "class", "interface", "extends",
        "implements", "static", "final", "abstract", "void", "int", "long",
        "double", "float", "boolean", "char", "byte", "short", "if", "else",
        "for", "while", "do", "switch", "case", "default", "break", "continue",
        "return", "try", "catch", "finally", "throw", "throws", "new", "this",
        "super", "import", "package", "true", "false", "null", "synchronized"
    )
    
    fun getKeywords(language: String): Set<String> = when (language.lowercase()) {
        "kotlin", "kt" -> KOTLIN_KEYWORDS
        "python", "py" -> PYTHON_KEYWORDS
        "javascript", "js", "typescript", "ts" -> JAVASCRIPT_KEYWORDS
        "java" -> JAVA_KEYWORDS
        else -> KOTLIN_KEYWORDS + PYTHON_KEYWORDS + JAVASCRIPT_KEYWORDS + JAVA_KEYWORDS
    }
}

/**
 * Parse and render markdown text with full styling
 */
@Composable
fun MarkdownText(
    text: String,
    modifier: Modifier = Modifier,
    isUserMessage: Boolean = false,
    onLinkClick: ((String) -> Unit)? = null
) {
    val clipboardManager = LocalClipboardManager.current
    val uriHandler = LocalUriHandler.current
    val scope = rememberCoroutineScope()
    
    // Clean the text first
    val cleanedText = remember(text) { TextCleaner.clean(text) }
    
    // Parse markdown blocks
    val blocks = remember(cleanedText) { parseMarkdownBlocks(cleanedText) }
    
    Column(modifier = modifier) {
        blocks.forEach { block ->
            when (block) {
                is MarkdownBlock.CodeBlock -> {
                    CodeBlockView(
                        code = block.code,
                        language = block.language,
                        onCopy = {
                            clipboardManager.setText(AnnotatedString(block.code))
                        }
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }
                is MarkdownBlock.Text -> {
                    val annotatedString = buildMarkdownAnnotatedString(
                        text = block.content,
                        isUserMessage = isUserMessage
                    )
                    
                    ClickableText(
                        text = annotatedString,
                        modifier = Modifier.fillMaxWidth(),
                        onClick = { offset ->
                            annotatedString.getStringAnnotations("URL", offset, offset)
                                .firstOrNull()?.let { annotation ->
                                    onLinkClick?.invoke(annotation.item) 
                                        ?: runCatching { uriHandler.openUri(annotation.item) }
                                }
                        }
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                }
            }
        }
    }
}

/**
 * Code block component with syntax highlighting
 */
@Composable
fun CodeBlockView(
    code: String,
    language: String,
    onCopy: () -> Unit
) {
    var showCopied by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(CodeTheme.Background)
    ) {
        // Header with language and copy button
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(CodeTheme.HeaderBackground)
                .padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = language.ifEmpty { "code" },
                color = CodeTheme.LineNumber,
                fontSize = 12.sp,
                fontFamily = FontFamily.Monospace
            )
            
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                if (showCopied) {
                    Text(
                        text = "Copied!",
                        color = CodeTheme.Function,
                        fontSize = 12.sp
                    )
                }
                
                IconButton(
                    onClick = {
                        onCopy()
                        showCopied = true
                        scope.launch {
                            kotlinx.coroutines.delay(2000)
                            showCopied = false
                        }
                    },
                    modifier = Modifier.size(28.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.ContentCopy,
                        contentDescription = "Copy code",
                        tint = CodeTheme.LineNumber,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }
        }
        
        // Code content with syntax highlighting
        SelectionContainer {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
                    .padding(12.dp)
            ) {
                val highlightedCode = buildSyntaxHighlightedText(code, language)
                Text(
                    text = highlightedCode,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 13.sp,
                    lineHeight = 20.sp
                )
            }
        }
    }
}

/**
 * Inline code component
 */
@Composable
fun InlineCode(
    text: String,
    modifier: Modifier = Modifier
) {
    Text(
        text = text,
        modifier = modifier
            .background(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(4.dp)
            )
            .padding(horizontal = 6.dp, vertical = 2.dp),
        fontFamily = FontFamily.Monospace,
        fontSize = 13.sp,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}

/**
 * Markdown block types
 */
sealed class MarkdownBlock {
    data class Text(val content: String) : MarkdownBlock()
    data class CodeBlock(val code: String, val language: String) : MarkdownBlock()
}

/**
 * Parse markdown into blocks
 */
private fun parseMarkdownBlocks(text: String): List<MarkdownBlock> {
    val blocks = mutableListOf<MarkdownBlock>()
    val codeBlockPattern = Regex("```(\\w*)\\n([\\s\\S]*?)```", RegexOption.MULTILINE)
    
    var lastIndex = 0
    codeBlockPattern.findAll(text).forEach { match ->
        // Add text before code block
        if (match.range.first > lastIndex) {
            val textBefore = text.substring(lastIndex, match.range.first).trim()
            if (textBefore.isNotEmpty()) {
                blocks.add(MarkdownBlock.Text(textBefore))
            }
        }
        
        // Add code block
        val language = match.groupValues[1]
        val code = match.groupValues[2].trim()
        blocks.add(MarkdownBlock.CodeBlock(code, language))
        
        lastIndex = match.range.last + 1
    }
    
    // Add remaining text
    if (lastIndex < text.length) {
        val remainingText = text.substring(lastIndex).trim()
        if (remainingText.isNotEmpty()) {
            blocks.add(MarkdownBlock.Text(remainingText))
        }
    }
    
    // If no blocks were created, treat entire text as plain text
    if (blocks.isEmpty() && text.isNotBlank()) {
        blocks.add(MarkdownBlock.Text(text))
    }
    
    return blocks
}

/**
 * Build syntax highlighted annotated string
 */
@Composable
private fun buildSyntaxHighlightedText(code: String, language: String): AnnotatedString {
    return buildAnnotatedString {
        val keywords = SyntaxPatterns.getKeywords(language)
        val lines = code.lines()
        
        lines.forEachIndexed { lineIndex, line ->
            highlightLine(line, keywords)
            if (lineIndex < lines.lastIndex) {
                append("\n")
            }
        }
    }
}

private fun AnnotatedString.Builder.highlightLine(line: String, keywords: Set<String>) {
    var i = 0
    
    while (i < line.length) {
        when {
            // Comments (// or #)
            (line.startsWith("//", i) || line.startsWith("#", i)) -> {
                val comment = line.substring(i)
                withStyle(SpanStyle(color = CodeTheme.Comment, fontStyle = FontStyle.Italic)) {
                    append(comment)
                }
                i = line.length
            }
            
            // String literals
            line[i] == '"' || line[i] == '\'' -> {
                val quote = line[i]
                val endIndex = findStringEnd(line, i, quote)
                val stringLiteral = line.substring(i, endIndex + 1)
                withStyle(SpanStyle(color = CodeTheme.String)) {
                    append(stringLiteral)
                }
                i = endIndex + 1
            }
            
            // Annotations (@Something)
            line[i] == '@' -> {
                val endIndex = findWordEnd(line, i + 1)
                val annotation = line.substring(i, endIndex)
                withStyle(SpanStyle(color = CodeTheme.Annotation)) {
                    append(annotation)
                }
                i = endIndex
            }
            
            // Numbers
            line[i].isDigit() -> {
                val endIndex = findNumberEnd(line, i)
                val number = line.substring(i, endIndex)
                withStyle(SpanStyle(color = CodeTheme.Number)) {
                    append(number)
                }
                i = endIndex
            }
            
            // Keywords and identifiers
            line[i].isLetter() || line[i] == '_' -> {
                val endIndex = findWordEnd(line, i)
                val word = line.substring(i, endIndex)
                
                val color = when {
                    keywords.contains(word) -> CodeTheme.Keyword
                    word.firstOrNull()?.isUpperCase() == true -> CodeTheme.Type
                    else -> CodeTheme.Text
                }
                
                // Check if it's a function call (followed by '(')
                val nextNonSpace = line.substring(endIndex).trimStart().firstOrNull()
                val actualColor = if (nextNonSpace == '(' && !keywords.contains(word)) {
                    CodeTheme.Function
                } else {
                    color
                }
                
                withStyle(SpanStyle(color = actualColor)) {
                    append(word)
                }
                i = endIndex
            }
            
            // Operators
            line[i] in "+-*/=<>!&|^~%:" -> {
                withStyle(SpanStyle(color = CodeTheme.Operator)) {
                    append(line[i])
                }
                i++
            }
            
            // Everything else
            else -> {
                withStyle(SpanStyle(color = CodeTheme.Text)) {
                    append(line[i])
                }
                i++
            }
        }
    }
}

private fun findStringEnd(line: String, start: Int, quote: Char): Int {
    var i = start + 1
    while (i < line.length) {
        if (line[i] == quote && line.getOrNull(i - 1) != '\\') {
            return i
        }
        i++
    }
    return line.length - 1
}

private fun findWordEnd(line: String, start: Int): Int {
    var i = start
    while (i < line.length && (line[i].isLetterOrDigit() || line[i] == '_')) {
        i++
    }
    return i
}

private fun findNumberEnd(line: String, start: Int): Int {
    var i = start
    while (i < line.length && (line[i].isDigit() || line[i] == '.' || line[i] == 'x' || 
            line[i] in 'a'..'f' || line[i] in 'A'..'F' || line[i] == 'L' || line[i] == 'f')) {
        i++
    }
    return i
}

/**
 * Build markdown annotated string with inline styling
 */
@Composable
private fun buildMarkdownAnnotatedString(
    text: String,
    isUserMessage: Boolean
): AnnotatedString {
    val textColor = if (isUserMessage) {
        MaterialTheme.colorScheme.onPrimary
    } else {
        MaterialTheme.colorScheme.onSurface
    }
    
    val linkColor = if (isUserMessage) {
        MaterialTheme.colorScheme.inversePrimary
    } else {
        MaterialTheme.colorScheme.primary
    }
    
    return buildAnnotatedString {
        withStyle(SpanStyle(color = textColor)) {
            var i = 0
            while (i < text.length) {
                when {
                    // Bold (**text**)
                    text.startsWith("**", i) -> {
                        val endIndex = text.indexOf("**", i + 2)
                        if (endIndex != -1) {
                            withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                                append(text.substring(i + 2, endIndex))
                            }
                            i = endIndex + 2
                        } else {
                            append("**")
                            i += 2
                        }
                    }
                    
                    // Italic (*text* or _text_)
                    (text[i] == '*' || text[i] == '_') && 
                    text.getOrNull(i + 1)?.let { it != '*' && it != '_' } == true -> {
                        val marker = text[i]
                        val endIndex = text.indexOf(marker, i + 1)
                        if (endIndex != -1 && endIndex > i + 1) {
                            withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                                append(text.substring(i + 1, endIndex))
                            }
                            i = endIndex + 1
                        } else {
                            append(text[i])
                            i++
                        }
                    }
                    
                    // Inline code (`code`)
                    text[i] == '`' && !text.startsWith("```", i) -> {
                        val endIndex = text.indexOf('`', i + 1)
                        if (endIndex != -1) {
                            withStyle(SpanStyle(
                                fontFamily = FontFamily.Monospace,
                                background = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                fontSize = 13.sp
                            )) {
                                append(" ${text.substring(i + 1, endIndex)} ")
                            }
                            i = endIndex + 1
                        } else {
                            append('`')
                            i++
                        }
                    }
                    
                    // Strikethrough (~~text~~)
                    text.startsWith("~~", i) -> {
                        val endIndex = text.indexOf("~~", i + 2)
                        if (endIndex != -1) {
                            withStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)) {
                                append(text.substring(i + 2, endIndex))
                            }
                            i = endIndex + 2
                        } else {
                            append("~~")
                            i += 2
                        }
                    }
                    
                    // Links [text](url)
                    text[i] == '[' -> {
                        val closeBracket = text.indexOf(']', i)
                        val openParen = closeBracket + 1
                        if (closeBracket != -1 && 
                            text.getOrNull(openParen) == '(' &&
                            text.indexOf(')', openParen) != -1) {
                            val closeParen = text.indexOf(')', openParen)
                            val linkText = text.substring(i + 1, closeBracket)
                            val url = text.substring(openParen + 1, closeParen)
                            
                            pushStringAnnotation("URL", url)
                            withStyle(SpanStyle(
                                color = linkColor,
                                textDecoration = TextDecoration.Underline
                            )) {
                                append(linkText)
                            }
                            pop()
                            i = closeParen + 1
                        } else {
                            append('[')
                            i++
                        }
                    }
                    
                    // Headers (# text)
                    text[i] == '#' && (i == 0 || text[i-1] == '\n') -> {
                        var level = 0
                        while (text.getOrNull(i + level) == '#') level++
                        
                        if (text.getOrNull(i + level) == ' ') {
                            val lineEnd = text.indexOf('\n', i + level + 1).let { 
                                if (it == -1) text.length else it 
                            }
                            val headerText = text.substring(i + level + 1, lineEnd)
                            
                            val fontSize = when (level) {
                                1 -> 24.sp
                                2 -> 20.sp
                                3 -> 18.sp
                                else -> 16.sp
                            }
                            
                            withStyle(SpanStyle(
                                fontWeight = FontWeight.Bold,
                                fontSize = fontSize
                            )) {
                                append(headerText)
                            }
                            i = lineEnd
                        } else {
                            append('#')
                            i++
                        }
                    }
                    
                    // Regular text
                    else -> {
                        append(text[i])
                        i++
                    }
                }
            }
        }
    }
}
