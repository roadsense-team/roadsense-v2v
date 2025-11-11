/**
 * @file Logger.h
 * @brief Simple serial logging utility
 *
 * Adapted from legacy SerialConsoleMiddleware
 * Provides colored console output with log levels (DEBUG, INFO, WARN, ERROR)
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <Arduino.h>

// ANSI color codes for terminal output
#define COLOR_RESET      "\x1b[0m"
#define COLOR_DEBUG      "\x1b[36m"    // Cyan
#define COLOR_INFO       "\x1b[32m"    // Green
#define COLOR_WARNING    "\x1b[33m"    // Yellow
#define COLOR_ERROR      "\x1b[31m"    // Red
#define COLOR_TIMESTAMP  "\x1b[90m"    // Gray

/**
 * @enum LogLevel
 * @brief Logging severity levels
 */
enum LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARNING = 2,
    LOG_ERROR = 3
};

/**
 * @class Logger
 * @brief Singleton logger for serial console output
 */
class Logger {
public:
    /**
     * @brief Get singleton instance
     */
    static Logger& getInstance();

    // Delete copy constructor and assignment
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief Initialize serial logging
     *
     * @param baud Serial baud rate (default 115200)
     */
    void begin(unsigned long baud = 115200);

    /**
     * @brief Log debug message
     *
     * @param module Module name (e.g., "GPS", "IMU", "Network")
     * @param message Message to log
     */
    void debug(const String& module, const String& message);

    /**
     * @brief Log info message
     *
     * @param module Module name
     * @param message Message to log
     */
    void info(const String& module, const String& message);

    /**
     * @brief Log warning message
     *
     * @param module Module name
     * @param message Message to log
     */
    void warning(const String& module, const String& message);

    /**
     * @brief Log error message
     *
     * @param module Module name
     * @param message Message to log
     */
    void error(const String& module, const String& message);

    /**
     * @brief Set minimum log level
     *
     * @param level Messages below this level will be ignored
     */
    void setLogLevel(LogLevel level);

    /**
     * @brief Enable/disable colored output
     *
     * @param useColor true to use ANSI colors, false for plain text
     */
    void setColoredOutput(bool useColor);

    /**
     * @brief Enable/disable timestamps
     *
     * @param show true to show timestamps, false to hide
     */
    void showTimestamps(bool show);

private:
    Logger();  // Private constructor for singleton

    void log(LogLevel level, const String& module, const String& message);
    String getLevelString(LogLevel level);
    const char* getLevelColor(LogLevel level);
    String getTimestamp();
    String colorize(const String& text, const char* colorCode);

    bool initialized;
    LogLevel minLevel;
    bool useColors;
    bool showTime;
    unsigned long baudRate;
};

// Global convenience function
inline Logger& Log() {
    return Logger::getInstance();
}

#endif // LOGGER_H
