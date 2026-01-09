package com.indxai.pilchat.util

import java.util.regex.Pattern

/**
 * Text Cleaner utility for cleaning model output
 * Removes unwanted characters like //%$ and other artifacts
 */
object TextCleaner {
    
    // Unicode escape pattern (matches \u followed by 4 hex digits)
    private val UNICODE_ESCAPE_PATTERN = Pattern.compile("\\\\u([0-9a-fA-F]{4})")
    
    // Patterns to clean from model output
    private val NOISE_PATTERNS = listOf(
        Regex("""//+%+\$*"""),               // Match //%$, //%%$, etc.
        Regex("""%%+"""),                     // Match %%, %%%, etc.
        Regex("""\$\$+"""),                   // Match $$, $$$, etc.
        Regex("""//+\s*$""", RegexOption.MULTILINE), // Trailing //
        Regex("""^\s*//+""", RegexOption.MULTILINE), // Leading //
        Regex("""\[\[+"""),                   // Match [[, [[[, etc.
        Regex("""\]\]+"""),                   // Match ]], ]]], etc.
        Regex("""<<+"""),                     // Match <<, <<<, etc.
        Regex(""">>+"""),                     // Match >>, >>>, etc.
        Regex("""~{3,}"""),                   // Match ~~~, ~~~~, etc.
        Regex("""={5,}"""),                   // Match =====, ======, etc.
        Regex("""\|{3,}"""),                  // Match |||, ||||, etc.
        Regex("""\\n{3,}"""),                 // Multiple escaped newlines
        Regex("""\x00-\x08"""),               // Control characters
        Regex("""\x0B\x0C"""),                // More control chars
        Regex("""\x0E-\x1F"""),               // Even more control chars
    )
    
    // Token artifacts from various models
    private val TOKEN_ARTIFACTS = listOf(
        "<|endoftext|>",
        "<|im_end|>",
        "<|im_start|>",
        "</s>",
        "<s>",
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "<pad>",
        "<unk>",
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
        "▁",  // SentencePiece underscore
    )
    
    // Repeated punctuation patterns
    private val REPEATED_PUNCT = Regex("""([.!?,;:])\1{2,}""")
    
    /**
     * Decode unicode escape sequences like \u201c to actual characters
     */
    private fun decodeUnicodeEscapes(text: String): String {
        val matcher = UNICODE_ESCAPE_PATTERN.matcher(text)
        val sb = StringBuffer()
        while (matcher.find()) {
            val hexCode = matcher.group(1) ?: continue
            try {
                val charCode = hexCode.toInt(16)
                matcher.appendReplacement(sb, charCode.toChar().toString())
            } catch (e: Exception) {
                // Keep original if decode fails
                matcher.appendReplacement(sb, matcher.group(0) ?: "")
            }
        }
        matcher.appendTail(sb)
        return sb.toString()
    }
    
    /**
     * Clean model output from artifacts and noise
     */
    fun clean(text: String): String {
        var cleaned = text
        
        // First decode unicode escape sequences (\u201c -> ", etc.)
        cleaned = decodeUnicodeEscapes(cleaned)
        
        // Replace smart quotes with regular quotes
        cleaned = cleaned
            .replace("\u201c", "\"")  // Left double quote
            .replace("\u201d", "\"")  // Right double quote
            .replace("\u2018", "'")   // Left single quote
            .replace("\u2019", "'")   // Right single quote
            .replace("\u00b7", "·")   // Middle dot
            .replace("\u2022", "•")   // Bullet
            .replace("\u2013", "–")   // En dash
            .replace("\u2014", "—")   // Em dash
            .replace("\u2026", "...") // Ellipsis
        
        // Remove token artifacts
        TOKEN_ARTIFACTS.forEach { artifact ->
            cleaned = cleaned.replace(artifact, "")
        }
        
        // Remove noise patterns
        NOISE_PATTERNS.forEach { pattern ->
            cleaned = cleaned.replace(pattern, "")
        }
        
        // Fix repeated punctuation (... is ok, but .... becomes ...)
        cleaned = cleaned.replace(REPEATED_PUNCT) { match ->
            val char = match.groupValues[1]
            if (char == ".") "..." else char  // Keep ellipsis as ...
        }
        
        // Clean up excessive whitespace
        cleaned = cleaned
            .replace(Regex("""\n{4,}"""), "\n\n\n")  // Max 3 newlines
            .replace(Regex("""[ \t]{3,}"""), "  ")    // Max 2 spaces
            .replace(Regex("""^\s+""", RegexOption.MULTILINE), "")  // Leading whitespace per line
        
        // Trim and return
        return cleaned.trim()
    }
    
    /**
     * Clean text for TTS (Text-to-Speech)
     * Removes markdown and formats for natural speech
     */
    fun cleanForTTS(text: String): String {
        var cleaned = clean(text)
        
        // Remove markdown formatting
        cleaned = cleaned
            .replace(Regex("""```[\s\S]*?```"""), "code block")  // Code blocks
            .replace(Regex("""`[^`]+`"""), "")                    // Inline code
            .replace(Regex("""\*\*([^*]+)\*\*"""), "$1")          // Bold
            .replace(Regex("""\*([^*]+)\*"""), "$1")              // Italic
            .replace(Regex("""__([^_]+)__"""), "$1")              // Bold alt
            .replace(Regex("""_([^_]+)_"""), "$1")                // Italic alt
            .replace(Regex("""~~([^~]+)~~"""), "$1")              // Strikethrough
            .replace(Regex("""^#+\s*""", RegexOption.MULTILINE), "") // Headers
            .replace(Regex("""^\s*[-*+]\s+""", RegexOption.MULTILINE), "") // List items
            .replace(Regex("""^\s*\d+\.\s+""", RegexOption.MULTILINE), "") // Numbered lists
            .replace(Regex("""!\[.*?\]\(.*?\)"""), "image")       // Images
            .replace(Regex("""\[([^\]]+)\]\([^)]+\)"""), "$1")    // Links
            .replace(Regex("""^>\s*""", RegexOption.MULTILINE), "") // Blockquotes
        
        // Replace URLs with "link"
        cleaned = cleaned.replace(
            Regex("""https?://[^\s]+"""), 
            "link"
        )
        
        // Clean up for natural speech
        cleaned = cleaned
            .replace("\n", ". ")
            .replace(Regex("""\.\s*\."""), ".")
            .replace(Regex("""\s+"""), " ")
            .trim()
        
        return cleaned
    }
    
    /**
     * Extract code blocks from text
     * Returns list of (language, code) pairs
     */
    fun extractCodeBlocks(text: String): List<Pair<String, String>> {
        val codeBlockPattern = Regex("""```(\w*)\n?([\s\S]*?)```""")
        return codeBlockPattern.findAll(text).map { match ->
            val language = match.groupValues[1].ifEmpty { "text" }
            val code = match.groupValues[2].trim()
            language to code
        }.toList()
    }
    
    /**
     * Check if text contains code blocks
     */
    fun hasCodeBlocks(text: String): Boolean {
        return text.contains(Regex("""```[\s\S]*?```"""))
    }
    
    /**
     * Check if text contains inline code
     */
    fun hasInlineCode(text: String): Boolean {
        return text.contains(Regex("""`[^`]+`"""))
    }
}
