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

	// Register tools
	enzu.NewTool(
		"WebSearch",
		"Performs web searches to gather market intelligence",
		func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("WebSearch requires 1 argument (search query)")
			}
			query, ok := args[0].(string)
			if !ok {
				return nil, fmt.Errorf("Search query must be a string")
			}
			return "Web search results for: " + query, nil
		},
		"WebSearchTools",
	)

	enzu.NewTool(
		"StrategyAnalysis",
		"Analyzes market data to develop AI-focused marketing strategies",
		func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("StrategyAnalysis requires 1 argument (market data)")
			}
			data, ok := args[0].(string)
			if !ok {
				return nil, fmt.Errorf("Market data must be a string")
			}
			return "Strategy analysis for: " + data, nil
		},
		"StrategyTools",
	)

	// Create agents
	marketAnalyst := enzu.NewAgent(
		"Lead Market Analyst",
		"Market Research Specialist",
		llm,
		enzu.WithToolLists("WebSearchTools"),
	)

	marketingStrategist := enzu.NewAgent(
		"Chief Marketing Strategist",
		"Marketing Strategy Expert",
		llm,
		enzu.WithToolLists("StrategyTools"),
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
