# Getting Started with Enzu

## Installation

First, install Enzu using Go's package manager:

```bash
go get github.com/teilomillet/enzu
```

Make sure you have an OpenAI API key set in your environment:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Basic Concepts

Enzu is built around these main concepts:

1. **Tools**: Functions that agents can use to perform actions
   - Custom tools for specific tasks
   - Built-in tools like ExaSearch
   - Tool inheritance and specialization

2. **Agents**: AI entities that can use tools and capabilities
   - Specialized or general-purpose
   - Parallel execution support
   - Tool inheritance control

3. **Tasks**: Units of work assigned to agents
   - Sequential or parallel execution
   - Dependencies between tasks
   - Clear, focused objectives

4. **Synergies**: Groups of agents working together
   - Tool sharing between agents
   - Task orchestration
   - Logging and monitoring

5. **SynergyManager**: Orchestrator for multiple synergies
   - Interactive mode support
   - Multi-team coordination
   - Result synthesis

## Advanced Features

### Parallel Execution

```go
// Create agent with parallel execution
agent := enzu.NewAgent(
    "FastAgent",
    "Processes tasks in parallel",
    llm,
    enzu.WithToolLists("BasicTools"),
    enzu.WithParallelExecution(true),
)
```

### Built-in Research Tools

```go
// Configure and register ExaSearch
options := tools.ExaSearchOptions{
    NumResults: 3,
    Type:       "neural",
    Contents: tools.Contents{
        Text: true,
    },
}
tools.ExaSearch("", options, "ResearchTool")
```

### Interactive Mode

```go
// Create interactive assistant
manager := enzu.NewSynergyManager(
    "Interactive Assistant",
    llm,
    logger,
)

// Add specialized synergies
manager.AddSynergy(researchSynergy)
manager.AddSynergy(analysisSynergy)

// Start interactive loop
fmt.Println("Ask me anything! Type 'exit' to quit.")
scanner := bufio.NewScanner(os.Stdin)
for scanner.Scan() {
    input := scanner.Text()
    if input == "exit" {
        break
    }
    results, _ := manager.ExecuteSynergies(
        context.Background(),
        input,
    )
    fmt.Printf("Response: %v\n", results["synthesis"])
}
```

## Your First Enzu Application

Let's create an example that demonstrates these concepts with a tool management system:

```go
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
    // Initialize LLM
    llm, err := gollm.NewLLM(
        gollm.SetProvider("openai"),
        gollm.SetModel("gpt-4o-mini"),
        gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
        gollm.SetMaxTokens(200),
        gollm.SetLogLevel(gollm.LogLevelDebug),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Create a logger
    logger := enzu.NewLogger(enzu.LogLevelDebug)

    // Create specialized tools
    enzu.NewTool(
        "CurrentTime",
        "Returns the current time",
        func(args ...interface{}) (interface{}, error) {
            return time.Now().Format(time.RFC3339), nil
        },
        "TimeTool", "BasicTools",  // Tool belongs to two lists
    )

    enzu.NewTool(
        "Add",
        "Adds two numbers",
        func(args ...interface{}) (interface{}, error) {
            if len(args) != 2 {
                return nil, fmt.Errorf("Add requires 2 arguments")
            }
            a, ok1 := args[0].(float64)
            b, ok2 := args[1].(float64)
            if !ok1 || !ok2 {
                return nil, fmt.Errorf("Arguments must be numbers")
            }
            return a + b, nil
        },
        "MathTool",
    )

    // Create specialized agents
    timeAgent := enzu.NewAgent(
        "Time Agent",
        "Handles time operations",
        llm,
        enzu.WithToolLists("TimeTool"),
    )

    mathAgent := enzu.NewAgent(
        "Math Agent",
        "Handles calculations",
        llm,
        enzu.WithToolLists("MathTool"),
    )

    // Create general agent that can use all tools
    generalAgent := enzu.NewAgent(
        "General Agent",
        "Can use all tools",
        llm,
        enzu.WithToolLists("BasicTools"),
        enzu.WithInheritSynergyTools(true),
    )

    // Create tasks
    timeTask := enzu.NewTask(
        "What time is it?",
        timeAgent,
    )

    mathTask := enzu.NewTask(
        "Add 5 and 7",
        mathAgent,
    )

    generalTask := enzu.NewTask(
        "Tell me the time and add some numbers",
        generalAgent,
    )

    // Create synergy
    synergy := enzu.NewSynergy(
        "Tool Demo",
        llm,
        enzu.WithAgents(timeAgent, mathAgent, generalAgent),
        enzu.WithTasks(timeTask, mathTask, generalTask),
        enzu.WithLogger(logger),
    )

    // Execute synergy
    ctx := context.Background()
    results, err := synergy.Execute(ctx)
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    for task, result := range results {
        fmt.Printf("Task: %s\nResult: %s\n\n", task, result)
    }
}

## Understanding the Code

1. **Tool Creation**:
   - Tools can belong to multiple lists (`TimeTool`, `BasicTools`)
   - Each tool has clear validation and error handling
   - Tools are organized by functionality

2. **Agent Specialization**:
   - Time Agent only uses time tools
   - Math Agent only uses math tools
   - General Agent can use all tools

3. **Tool Inheritance**:
   - General Agent inherits tools from synergy
   - Specialized agents only use their assigned tools

4. **Task Organization**:
   - Each task is assigned to the appropriate agent
   - Tasks are clear and focused

## Next Steps

1. Learn about parallel execution in the [Core Concepts](./core-concepts.md#parallel-execution) guide
2. Try the built-in tools in the [Search Example](../examples/5_search_example.go)
3. Build an interactive assistant using the [Manager Mode Example](../examples/7_manager_mode_example.go)

## Common Issues

1. **API Key**: Ensure your OpenAI API key is set correctly
2. **Tool Lists**: Make sure tool lists exist before assigning to agents
3. **Task Dependencies**: Consider task order when using dependencies
4. **Parallel Execution**: Monitor resource usage with parallel tasks
5. **Interactive Mode**: Handle user input validation and error cases

For more examples and detailed documentation, visit our [GitHub repository](https://github.com/teilomillet/enzu).
