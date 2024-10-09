package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/teilomillet/enzu"
	"github.com/teilomillet/gollm"
)

func main() {
	// Load API key from environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create a new LLM instance with custom configuration
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(200),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(time.Second*2),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to initialize LLM: %v", err)
	}

	// Create a logger for Enzu
	logger := enzu.NewLogger(enzu.LogLevelWarn)

	// Create agents
	marketAnalyst := enzu.NewAgent(
		"Lead Market Analyst",
		"Market Research Specialist",
		llm,
		enzu.WithCapabilities(&WebSearchCapability{}),
	)

	marketingStrategist := enzu.NewAgent(
		"Chief Marketing Strategist",
		"Marketing Strategy Expert",
		llm,
		enzu.WithEnhancement(&enzu.EnhancementConfig{
			Instructions: "Focus on innovative AI-driven marketing strategies",
		}),
	)

	contentCreator := enzu.NewAgent(
		"Creative Content Creator",
		"Content Creation Specialist",
		llm,
	)

	// Create tasks
	researchTask := enzu.NewTask(
		"Analyze current market trends for AI solutions, focusing on multi-agent systems like CrewAI",
		marketAnalyst,
	)

	strategyTask := enzu.NewTask(
		"Develop a marketing strategy for CrewAI, highlighting its unique features and benefits",
		marketingStrategist,
		researchTask,
	)

	contentTask := enzu.NewTask(
		"Create marketing copy for a new CrewAI campaign, emphasizing its advantages over traditional AI solutions",
		contentCreator,
		strategyTask,
	)

	// Create synergy with logger
	marketingSynergy := enzu.NewSynergy(
		"Develop and execute a marketing campaign for CrewAI",
		llm,
		enzu.WithAgents(marketAnalyst, marketingStrategist, contentCreator),
		enzu.WithTasks(researchTask, strategyTask, contentTask),
		enzu.WithLogger(logger),
	)

	// Execute the synergy
	ctx := context.Background()
	results, err := marketingSynergy.Execute(ctx)
	if err != nil {
		log.Fatalf("Error executing synergy: %v", err)
	}

	// Print results
	for task, result := range results {
		fmt.Printf("Task: %s\nResult: %s\n\n", task, result)
	}
}

// WebSearchCapability is a simple implementation of the Capability interface
type WebSearchCapability struct{}

func (w *WebSearchCapability) Execute(ctx context.Context, input string) (string, error) {
	// Implement web search logic here
	return "Web search results for: " + input, nil
}
