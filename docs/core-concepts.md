# Core Concepts

This guide explains the core concepts of Enzu in detail, with examples for each concept.

## Tools

Tools are functions that agents can use to take actions. They are the building blocks of agent capabilities.

### Anatomy of a Tool

```go
// Basic tool with single tool list
enzu.NewTool(
    "ReadFile",                              // Name
    "Reads content from a file",             // Description
    func(args ...interface{}) (interface{}, error) {
        if len(args) != 1 {
            return nil, fmt.Errorf("ReadFile requires 1 argument (filepath)")
        }
        filepath, ok := args[0].(string)
        if !ok {
            return nil, fmt.Errorf("Argument must be a string")
        }
        content, err := ioutil.ReadFile(filepath)
        return string(content), err
    },
    "FileTools",                             // Tool list name
)

// Tool belonging to multiple tool lists
enzu.NewTool(
    "CurrentTime",
    "Returns the current time",
    func(args ...interface{}) (interface{}, error) {
        return time.Now().Format(time.RFC3339), nil
    },
    "TimeTool", "BasicTools",  // Multiple tool lists
)
```

### Tool Organization

Tools can be organized into ToolLists for better management and specialization:

```go
// Create specialized tools
enzu.NewTool("Add", "Adds numbers", addFn, "MathTool")
enzu.NewTool("Subtract", "Subtracts numbers", subtractFn, "MathTool")
enzu.NewTool("CurrentTime", "Gets current time", timeFn, "TimeTool")
enzu.NewTool("ReadFile", "Reads files", readFn, "FileTools")

// Create specialized agents
mathAgent := enzu.NewAgent(
    "Math Agent",
    "Handles mathematical operations",
    llm,
    enzu.WithToolLists("MathTool"),
)

timeAgent := enzu.NewAgent(
    "Time Agent",
    "Handles time-related operations",
    llm,
    enzu.WithToolLists("TimeTool"),
)

// Create general-purpose agent with all tools
generalAgent := enzu.NewAgent(
    "General Agent",
    "Has access to all tools",
    llm,
    enzu.WithToolLists("MathTool", "TimeTool", "FileTools"),
)
```

### Tool Inheritance

Agents can inherit tools from their synergy:

```go
// Agent that inherits synergy tools (default behavior)
agent1 := enzu.NewAgent(
    "Inheriting Agent",
    "Can use synergy tools",
    llm,
    enzu.WithToolLists("MathTool"),
    enzu.WithInheritSynergyTools(true),
)

// Agent that doesn't inherit synergy tools
agent2 := enzu.NewAgent(
    "Restricted Agent",
    "Only uses own tools",
    llm,
    enzu.WithToolLists("MathTool"),
    enzu.WithInheritSynergyTools(false),
)

// Create synergy with shared tools
synergy := enzu.NewSynergy(
    "Tool Inheritance Demo",
    llm,
    enzu.WithAgents(agent1, agent2),
    enzu.WithTools("TimeTool"),  // Only agent1 can use these
)
```

## Agents

Agents are AI entities that can perform specific tasks using tools, capabilities, and enhancements. Think of them as specialized workers in your system.

### Creating an Agent

```go
// Basic agent with tools
agent := enzu.NewAgent(
    "DataAnalyst",                           // Name
    "Analyzes data and generates reports",   // Description
    llm,                                     // LLM instance
    enzu.WithToolLists("AnalysisTools"),     // Tools the agent can use
)

// Agent with capabilities
agent := enzu.NewAgent(
    "WebResearcher",
    "Performs web research",
    llm,
    enzu.WithCapabilities(&WebSearchCapability{}),
)

// Agent with enhancements
agent := enzu.NewAgent(
    "Strategist",
    "Develops strategies",
    llm,
    enzu.WithEnhancement(&enzu.EnhancementConfig{
        Instructions: "Focus on innovative AI-driven strategies",
    }),
)
```

### Agent Properties

- **Name**: Identifies the agent
- **Description**: Explains the agent's purpose and capabilities
- **Tools**: Set of tools the agent can use
- **Capabilities**: Custom behaviors that extend agent functionality
- **Enhancements**: Configuration to modify agent behavior

## Tasks

Tasks are units of work assigned to agents. They can be executed independently or in sequence.

### Creating Tasks

```go
// Simple task
task := enzu.NewTask(
    "Read config.json and post its contents to API",
    fileAgent,
)

// Sequential tasks
researchTask := enzu.NewTask(
    "Analyze market trends",
    marketAnalyst,
)

strategyTask := enzu.NewTask(
    "Develop marketing strategy",
    marketingStrategist,
    researchTask,  // This task depends on researchTask
)
```

### Task Properties

- **Description**: What needs to be done
- **Agent**: The agent responsible for the task
- **Dependencies**: Other tasks that must complete first

## Synergies

Synergies orchestrate multiple agents and tasks to achieve complex goals. They can be managed individually or through a SynergyManager.

### Basic Synergy

```go
synergy := enzu.NewSynergy(
    "Marketing Campaign",
    llm,
    enzu.WithAgents(marketAnalyst, strategist),
    enzu.WithTasks(researchTask, strategyTask),
    enzu.WithLogger(logger),
)
```

### SynergyManager

For complex projects with multiple synergies:

```go
// Create SynergyManager
manager := enzu.NewSynergyManager("MainManager", llm, logger)

// Create and add marketing team synergy
marketingAgents := []*enzu.Agent{
    enzu.NewAgent("Market Analyst", "Analyze trends", llm),
    enzu.NewAgent("Strategist", "Create strategy", llm),
}
marketingSynergy := enzu.NewSynergy(
    "Marketing Strategy",
    llm,
    enzu.WithAgents(marketingAgents...),
    enzu.WithTasks(
        enzu.NewTask("Analyze trends", marketingAgents[0]),
        enzu.NewTask("Create strategy", marketingAgents[1]),
    ),
)
manager.AddSynergy(marketingSynergy)

// Create and add product team synergy
productAgents := []*enzu.Agent{
    enzu.NewAgent("Product Manager", "Define features", llm),
    enzu.NewAgent("Tech Lead", "Create roadmap", llm),
}
productSynergy := enzu.NewSynergy(
    "Product Development",
    llm,
    enzu.WithAgents(productAgents...),
    enzu.WithTasks(
        enzu.NewTask("Define features", productAgents[0]),
        enzu.NewTask("Create roadmap", productAgents[1]),
    ),
)
manager.AddSynergy(productSynergy)

// Execute all synergies with a single prompt
results, err := manager.ExecuteSynergies(ctx, 
    "Plan launch of new AI product")
```

## Parallel Execution

Agents can execute tasks in parallel for improved performance:

```go
// Create agent with parallel execution enabled
agent := enzu.NewAgent(
    "ParallelAgent",
    "Executes tasks in parallel",
    llm,
    enzu.WithToolLists("BasicTools"),
    enzu.WithParallelExecution(true),  // Enable parallel execution
)

// Create multiple tasks
task1 := enzu.NewTask("Task 1", agent)
task2 := enzu.NewTask("Task 2", agent)

// Tasks will execute in parallel
synergy := enzu.NewSynergy(
    "Parallel Processing",
    llm,
    enzu.WithAgents(agent),
    enzu.WithTasks(task1, task2),
)
```

## Built-in Tools

Enzu provides built-in tools for common operations:

### ExaSearch Tool

```go
// Configure ExaSearch options
options := tools.ExaSearchOptions{
    NumResults: 5,
    Type:       "neural",
    Contents: tools.Contents{
        Text: true,
    },
    UseAutoprompt:      true,
    StartPublishedDate: "2023-01-01T00:00:00.000Z",
}

// Register ExaSearch tool
tools.ExaSearch("", options, "ResearchTool")

// Create research agent
researchAgent := enzu.NewAgent(
    "Researcher",
    "Conducts web research",
    llm,
    enzu.WithToolLists("ResearchTool"),
)
```

## Interactive Mode

Enzu supports interactive mode for building conversational AI assistants:

```go
// Create SynergyManager for interactive mode
manager := enzu.NewSynergyManager(
    "Interactive Assistant",
    llm,
    logger,
)

// Add specialized synergies
manager.AddSynergy(createResearchSynergy(llm, logger))
manager.AddSynergy(createAnalysisSynergy(llm, logger))
manager.AddSynergy(createCreativeSynergy(llm, logger))

// Handle user input
func handleUserInput(manager *enzu.SynergyManager, input string) {
    ctx := context.Background()
    results, err := manager.ExecuteSynergies(ctx, input)
    if err != nil {
        log.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Response: %v\n", results["synthesis"])
}
```

### Creating Specialized Synergies

```go
func createResearchSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
    agent := enzu.NewAgent(
        "Research Agent",
        "Conducts research",
        llm,
        enzu.WithToolLists("ResearchTool"),
    )
    
    return enzu.NewSynergy(
        "Research Operations",
        llm,
        enzu.WithAgents(agent),
        enzu.WithLogger(logger),
    )
}
```

## Best Practices

1. **Tool Organization**
   - Group related tools into focused tool lists
   - Consider tool inheritance when designing agent access
   - Use clear error messages in tool implementations

2. **Agent Design**
   - Create specialized agents for specific domains
   - Use tool inheritance thoughtfully
   - Combine capabilities and enhancements as needed

3. **Synergy Management**
   - Use SynergyManager for complex, multi-team projects
   - Keep synergies focused on specific objectives
   - Consider tool sharing between agents carefully

4. **Error Handling**
   - Implement comprehensive error checking in tools
   - Use logging to track execution
   - Handle tool failures gracefully

5. **Parallel Processing**
   - Use parallel execution for independent tasks
   - Consider resource constraints
   - Handle concurrent tool access properly

6. **Interactive Mode**
   - Define clear capabilities
   - Create specialized synergies for different functions
   - Maintain conversation context
   - Handle errors gracefully

## Complete Example: Data Processing Pipeline

Here's a complete example that combines all concepts:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/teilomillet/enzu"
    "github.com/teilomillet/gollm"
)

func main() {
    // Initialize LLM
    llm := initLLM()

    // Create tools
    createFileTools()
    createProcessingTools()
    createOutputTools()

    // Create agents
    reader := enzu.NewAgent("Reader", "Reads input files", llm,
        enzu.WithToolLists("FileTools"))
    
    processor := enzu.NewAgent("Processor", "Processes data", llm,
        enzu.WithToolLists("ProcessingTools"))
    
    writer := enzu.NewAgent("Writer", "Writes results", llm,
        enzu.WithToolLists("OutputTools"))

    // Create tasks
    readTask := enzu.NewTask("Read input.csv", reader)
    processTask := enzu.NewTask("Process the data", processor)
    writeTask := enzu.NewTask("Write results to output.json", writer)

    // Create synergy
    synergy := enzu.NewSynergy(
        "Data Pipeline",
        llm,
        enzu.WithAgents(reader, processor, writer),
        enzu.WithTasks(readTask, processTask, writeTask),
        enzu.WithParallel(false),  // Sequential execution
    )

    // Execute
    results, err := synergy.Execute(context.Background())
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Pipeline completed: %v\n", results)
}

func createFileTools() {
    enzu.NewTool("ReadCSV", "Reads CSV file", readCSVFn, "FileTools")
    // Add more file tools...
}

func createProcessingTools() {
    enzu.NewTool("ProcessData", "Processes data", processDataFn, "ProcessingTools")
    // Add more processing tools...
}

func createOutputTools() {
    enzu.NewTool("WriteJSON", "Writes JSON file", writeJSONFn, "OutputTools")
    // Add more output tools...
}
```

This example shows how to:
1. Organize tools into logical groups
2. Create specialized agents
3. Define a sequence of tasks
4. Orchestrate everything with a synergy

Continue to the [Tutorials](./tutorials/README.md) section for practical examples of these concepts in action.

For complete examples, see our [examples directory](../examples):
- [Basic Example](../examples/1_basic_example.go)
- [Manager Example](../examples/2_manager_example.go)
- [Tools Example](../examples/3_tools_example.go)
- [Parallel Example](../examples/4_parallel_example.go)
- [Search Example](../examples/5_search_example.go)
- [Manager Mode Example](../examples/7_manager_mode_example.go)
