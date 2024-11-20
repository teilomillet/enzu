// Package enzu provides logging functionality for the Enzu framework.
// This file implements a flexible, leveled logging system that supports
// categorized log messages with different severity levels and formatted output.
package enzu

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

// LogLevel represents the severity level of a log message.
// Higher values indicate more verbose logging, with Debug being
// the most verbose and Off completely disabling logging.
type LogLevel int

// Log level constants define the available logging severity levels.
// These levels follow common logging conventions:
//   - LogLevelOff: Disables all logging
//   - LogLevelError: Only critical errors
//   - LogLevelWarn: Warnings and errors
//   - LogLevelInfo: General information, warnings, and errors
//   - LogLevelDebug: Detailed debug information and all above levels
const (
	LogLevelOff LogLevel = iota
	LogLevelError
	LogLevelWarn
	LogLevelInfo
	LogLevelDebug
)

// Logger provides structured logging capabilities for the Enzu framework.
// It supports different log levels, categorized messages, and formatted output
// with timestamps. The logger can be configured to show or hide messages based
// on their severity level.
type Logger struct {
	// level determines which messages are logged based on their severity
	level LogLevel
	
	// logger is the underlying Go standard library logger
	logger *log.Logger
}

// NewLogger creates a new Logger instance with the specified log level.
// The logger writes to standard output with a custom format that includes
// timestamps, log levels, and message categories.
//
// Parameters:
//   - level: The minimum severity level of messages to log
//
// Returns:
//   - *Logger: A new logger instance configured with the specified level
func NewLogger(level LogLevel) *Logger {
	return &Logger{
		level:  level,
		logger: log.New(os.Stdout, "", 0),
	}
}

// SetLevel changes the logger's minimum severity level.
// Messages with a severity level lower than this will not be logged.
//
// Parameters:
//   - level: The new minimum severity level for logging
func (l *Logger) SetLevel(level LogLevel) {
	l.level = level
}

// log is an internal method that handles the actual logging of messages.
// It formats the message with a timestamp, level indicator, and category,
// and writes it to the output if the message's level is within the logger's
// configured severity threshold.
//
// Parameters:
//   - level: Severity level of the message
//   - category: Category or component the message relates to
//   - message: Format string for the log message
//   - args: Arguments to be formatted into the message
func (l *Logger) log(level LogLevel, category, message string, args ...interface{}) {
	if level <= l.level {
		timestamp := time.Now().Format("15:04:05")
		levelStr := strings.ToUpper(level.String())
		formattedMessage := fmt.Sprintf(message, args...)

		l.logger.Printf("\n%s [%-5s] %-15s\n%s\n%s\n",
			timestamp, levelStr, category,
			strings.Repeat("-", 50),
			formattedMessage)
	}
}

// Debug logs a message at DEBUG level.
// These messages are typically used during development and debugging
// to trace program execution and internal state.
//
// Parameters:
//   - category: The component or area of the code generating the message
//   - message: Format string for the log message
//   - args: Values to be formatted into the message
func (l *Logger) Debug(category, message string, args ...interface{}) {
	l.log(LogLevelDebug, category, message, args...)
}

// Info logs a message at INFO level.
// These messages provide general information about program execution
// that could be helpful to users and administrators.
//
// Parameters:
//   - category: The component or area of the code generating the message
//   - message: Format string for the log message
//   - args: Values to be formatted into the message
func (l *Logger) Info(category, message string, args ...interface{}) {
	l.log(LogLevelInfo, category, message, args...)
}

// Warn logs a message at WARN level.
// These messages indicate potentially harmful situations or
// unexpected states that the program can recover from.
//
// Parameters:
//   - category: The component or area of the code generating the message
//   - message: Format string for the log message
//   - args: Values to be formatted into the message
func (l *Logger) Warn(category, message string, args ...interface{}) {
	l.log(LogLevelWarn, category, message, args...)
}

// Error logs a message at ERROR level.
// These messages indicate serious problems that need
// immediate attention from users or administrators.
//
// Parameters:
//   - category: The component or area of the code generating the message
//   - message: Format string for the log message
//   - args: Values to be formatted into the message
func (l *Logger) Error(category, message string, args ...interface{}) {
	l.log(LogLevelError, category, message, args...)
}

// String converts a LogLevel to its string representation.
// This method is used internally for formatting log messages
// and implements the Stringer interface.
//
// Returns:
//   - string: The string representation of the log level (e.g., "DEBUG", "INFO")
func (l LogLevel) String() string {
	return [...]string{"OFF", "ERROR", "WARN", "INFO", "DEBUG"}[l]
}
