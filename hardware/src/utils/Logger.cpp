/**
 * @file Logger.cpp
 * @brief Implementation of Logger utility
 */

#include "Logger.h"

Logger::Logger() : initialized(false), minLevel(LOG_DEBUG),
                   useColors(true), showTime(true), baudRate(115200) {}

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::begin(unsigned long baud) {
    if (!initialized) {
        Serial.begin(baud);
        baudRate = baud;
        initialized = true;

        // Wait for serial to initialize
        delay(100);

        // Print startup message
        Serial.println();
        Serial.println(colorize("=================================", COLOR_INFO));
        Serial.println(colorize("  RoadSense V2V Logger Started  ", COLOR_INFO));
        Serial.println(colorize("=================================", COLOR_INFO));
        Serial.println();
    }
}

void Logger::debug(const String& module, const String& message) {
    log(LOG_DEBUG, module, message);
}

void Logger::info(const String& module, const String& message) {
    log(LOG_INFO, module, message);
}

void Logger::warning(const String& module, const String& message) {
    log(LOG_WARNING, module, message);
}

void Logger::error(const String& module, const String& message) {
    log(LOG_ERROR, module, message);
}

void Logger::setLogLevel(LogLevel level) {
    minLevel = level;
}

void Logger::setColoredOutput(bool useColor) {
    useColors = useColor;
}

void Logger::showTimestamps(bool show) {
    showTime = show;
}

void Logger::log(LogLevel level, const String& module, const String& message) {
    if (!initialized || level < minLevel) {
        return;
    }

    String output = "";

    // Add timestamp
    if (showTime) {
        output += colorize(getTimestamp(), COLOR_TIMESTAMP);
        output += " ";
    }

    // Add level
    String levelStr = "[" + getLevelString(level) + "]";
    output += colorize(levelStr, getLevelColor(level));
    output += " ";

    // Add module
    if (module.length() > 0) {
        output += "[" + module + "] ";
    }

    // Add message
    output += message;

    Serial.println(output);
}

String Logger::getLevelString(LogLevel level) {
    switch (level) {
        case LOG_DEBUG:   return "DEBUG";
        case LOG_INFO:    return "INFO ";
        case LOG_WARNING: return "WARN ";
        case LOG_ERROR:   return "ERROR";
        default:          return "?????";
    }
}

const char* Logger::getLevelColor(LogLevel level) {
    switch (level) {
        case LOG_DEBUG:   return COLOR_DEBUG;
        case LOG_INFO:    return COLOR_INFO;
        case LOG_WARNING: return COLOR_WARNING;
        case LOG_ERROR:   return COLOR_ERROR;
        default:          return COLOR_RESET;
    }
}

String Logger::getTimestamp() {
    unsigned long ms = millis();
    unsigned long seconds = ms / 1000;
    unsigned long minutes = seconds / 60;
    unsigned long hours = minutes / 60;

    ms %= 1000;
    seconds %= 60;
    minutes %= 60;
    hours %= 24;

    char buffer[16];
    snprintf(buffer, sizeof(buffer), "%02lu:%02lu:%02lu.%03lu",
             hours, minutes, seconds, ms);
    return String(buffer);
}

String Logger::colorize(const String& text, const char* colorCode) {
    if (!useColors) {
        return text;
    }
    return String(colorCode) + text + String(COLOR_RESET);
}
