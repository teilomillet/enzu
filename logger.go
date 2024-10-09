package enzu

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

type LogLevel int

const (
	LogLevelOff LogLevel = iota
	LogLevelError
	LogLevelWarn
	LogLevelInfo
	LogLevelDebug
)

type Logger struct {
	level  LogLevel
	logger *log.Logger
}

func NewLogger(level LogLevel) *Logger {
	return &Logger{
		level:  level,
		logger: log.New(os.Stdout, "", 0),
	}
}

func (l *Logger) SetLevel(level LogLevel) {
	l.level = level
}

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

func (l *Logger) Debug(category, message string, args ...interface{}) {
	l.log(LogLevelDebug, category, message, args...)
}

func (l *Logger) Info(category, message string, args ...interface{}) {
	l.log(LogLevelInfo, category, message, args...)
}

func (l *Logger) Warn(category, message string, args ...interface{}) {
	l.log(LogLevelWarn, category, message, args...)
}

func (l *Logger) Error(category, message string, args ...interface{}) {
	l.log(LogLevelError, category, message, args...)
}

func (l LogLevel) String() string {
	return [...]string{"OFF", "ERROR", "WARN", "INFO", "DEBUG"}[l]
}
