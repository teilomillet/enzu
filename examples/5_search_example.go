package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/enzu"
	"github.com/teilomillet/enzu/tools"
	"github.com/teilomillet/gollm"
)

func main() {
	// Load API key from environment variable
	openAIKey := os.Getenv("OPENAI_API_KEY")
	if openAIKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create a new LLM instance
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(openAIKey),
		gollm.SetMaxTokens(200),
	)
	if err != nil {
		log.Fatalf("Failed to initialize LLM: %v", err)
	}

	// Create a logger for Enzu
	logger := enzu.NewLogger(enzu.LogLevelInfo)

	// Define options for the ExaSearch tool
	exaSearchOptions := tools.ExaSearchOptions{
		NumResults: 5,
		Type:       "neural",
		Contents: tools.Contents{
			Text: true,
		},
		UseAutoprompt:      true,
		StartPublishedDate: "2023-01-01T00:00:00.000Z",
	}

	// Register the ExaSearch tool
	tools.ExaSearch("", exaSearchOptions, "ResearchTool")

	// Create research agents
	researchAgent1 := enzu.NewAgent("Research Agent 1", "Agent specialized in AI research", llm,
		enzu.WithToolLists("ResearchTool"),
		enzu.WithParallelExecution(true),
	)
	researchAgent2 := enzu.NewAgent("Research Agent 2", "Agent specialized in startup research", llm,
		enzu.WithToolLists("ResearchTool"),
		enzu.WithParallelExecution(true),
	)

	// Create research tasks
	task1 := enzu.NewTask("Research the latest advancements in artificial intelligence", researchAgent1)
	task2 := enzu.NewTask("Find information about the most promising AI startups in 2024", researchAgent2)

	// Create a synergy
	synergy := enzu.NewSynergy(
		"Parallel AI Research",
		llm,
		enzu.WithAgents(researchAgent1, researchAgent2),
		enzu.WithTasks(task1, task2),
		enzu.WithLogger(logger),
	)

	// Execute the synergy
	ctx := context.Background()
	results, err := synergy.Execute(ctx)
	if err != nil {
		log.Fatalf("Error executing synergy: %v", err)
	}

	// Print results
	fmt.Println("Research Results:")
	for task, result := range results {
		fmt.Printf("Task: %s\nResult: %v\n\n", task, result)
	}
}
