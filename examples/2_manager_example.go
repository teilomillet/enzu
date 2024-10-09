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
	logger := enzu.NewLogger(enzu.LogLevelDebug)

	// Create SynergyManager
	manager := enzu.NewSynergyManager("MainManager", llm, logger)

	// Create first Synergy (Marketing Team)
	marketingAgents := createMarketingTeam(llm)
	marketingSynergy := enzu.NewSynergy(
		"Develop Marketing Strategy",
		llm,
		enzu.WithAgents(marketingAgents...),
		enzu.WithTasks(
			enzu.NewTask("Analyze current market trends", marketingAgents[0]),
			enzu.NewTask("Develop marketing strategy", marketingAgents[1]),
			enzu.NewTask("Create marketing copy", marketingAgents[2]),
		),
		enzu.WithLogger(logger),
	)
	manager.AddSynergy(marketingSynergy)

	// Create second Synergy (Product Development Team)
	productAgents := createProductTeam(llm)
	productSynergy := enzu.NewSynergy(
		"Develop New Product Features",
		llm,
		enzu.WithAgents(productAgents...),
		enzu.WithTasks(
			enzu.NewTask("Brainstorm new features", productAgents[0]),
			enzu.NewTask("Prioritize features", productAgents[1]),
			enzu.NewTask("Create development roadmap", productAgents[2]),
		),
		enzu.WithLogger(logger),
	)
	manager.AddSynergy(productSynergy)

	// Execute all Synergies
	ctx := context.Background()
	initialPrompt := "Develop a comprehensive plan to launch a new AI-powered project management tool"
	results, err := manager.ExecuteSynergies(ctx, initialPrompt)
	if err != nil {
		log.Fatalf("Error executing Synergies: %v", err)
	}

	// Print synthesized results
	fmt.Printf("Synthesized Results:\n%v\n", results["synthesis"])
}

func createMarketingTeam(llm gollm.LLM) []*enzu.Agent {
	return []*enzu.Agent{
		enzu.NewAgent("Market Analyst", "Analyze market trends", llm),
		enzu.NewAgent("Marketing Strategist", "Develop marketing strategies", llm),
		enzu.NewAgent("Content Creator", "Create marketing content", llm),
	}
}

func createProductTeam(llm gollm.LLM) []*enzu.Agent {
	return []*enzu.Agent{
		enzu.NewAgent("Product Visionary", "Brainstorm new features", llm),
		enzu.NewAgent("Product Manager", "Prioritize features", llm),
		enzu.NewAgent("Tech Lead", "Create development roadmap", llm),
	}
}
