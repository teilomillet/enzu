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

	// Create a new LLM instance
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(200),
		gollm.SetLogLevel(gollm.LogLevelDebug),
	)
	if err != nil {
		log.Fatalf("Failed to initialize LLM: %v", err)
	}

	// Create a logger for Enzu
	logger := enzu.NewLogger(enzu.LogLevelDebug)

	// Create tools
	enzu.NewTool(
		"CurrentTime",
		"Returns the current time",
		func(args ...interface{}) (interface{}, error) {
			return time.Now().Format(time.RFC3339), nil
		},
		"TimeTool", "BasicTools",
	)

	enzu.NewTool(
		"Add",
		"Adds two numbers",
		func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("Add tool requires exactly 2 arguments")
			}
			a, ok1 := args[0].(float64)
			b, ok2 := args[1].(float64)
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("Add tool arguments must be numbers")
			}
			return a + b, nil
		},
		"MathTool", "BasicTools",
	)

	// Create agents
	timeAgent := enzu.NewAgent("Time Agent", "Agent specialized in time-related tasks", llm,
		enzu.WithToolLists("TimeTool"),
		enzu.WithParallelExecution(true), // This agent will execute tasks in parallel
	)
	mathAgent := enzu.NewAgent("Math Agent", "Agent specialized in mathematical operations", llm,
		enzu.WithToolLists("MathTool"),
	)
	generalAgent := enzu.NewAgent("General Agent", "Agent with access to all tools", llm,
		enzu.WithToolLists("BasicTools"),
		enzu.WithInheritSynergyTools(true), // Explicitly set to true (default behavior)
	)
	restrictedAgent := enzu.NewAgent("Restricted Agent", "Agent with no inheritance from Synergy", llm,
		enzu.WithToolLists("MathTool"),
		enzu.WithInheritSynergyTools(false), // This agent will not inherit Synergy tools
	)

	// Create tasks
	timeTask := enzu.NewTask("What is the current time?", timeAgent)
	mathTask := enzu.NewTask("Add 5 and 7", mathAgent)
	combinedTask := enzu.NewTask("What is the current time and what is 5 + 7?", generalAgent)
	restrictedTask := enzu.NewTask("Add 10 and 20, and try to get the current time", restrictedAgent)

	// Create a synergy with a new ToolRegistry and the logger
	synergy := enzu.NewSynergy(
		"Demonstrate tool usage, agent specialization, and inheritance control",
		llm,
		enzu.WithAgents(timeAgent, mathAgent, generalAgent, restrictedAgent),
		enzu.WithTasks(timeTask, mathTask, combinedTask, restrictedTask),
		enzu.WithTools("TimeTool"), // This makes only TimeTool available to the synergy
		enzu.WithLogger(logger),    // Use the created logger
	)

	// Execute the synergy
	ctx := context.Background()
	results, err := synergy.Execute(ctx)
	if err != nil {
		log.Fatalf("Error executing synergy: %v", err)
	}

	// Print results
	fmt.Println("Synergy Results:")
	for task, result := range results {
		fmt.Printf("Task: %s\nResult: %s\n\n", task, result)
	}
}
