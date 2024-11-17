# Building an Interactive AI Assistant with Manager Mode

This tutorial demonstrates how to build a sophisticated interactive AI assistant using Enzu's manager mode. We'll create an assistant that can handle research, analysis, and creative tasks through specialized synergies.

## Overview

The manager mode allows you to:
1. Create multiple specialized synergies for different capabilities
2. Handle interactive conversations with users
3. Maintain conversation history for context
4. Route tasks to appropriate synergies based on user input

## Prerequisites

- Go 1.16 or later
- OpenAI API key
- Enzu framework
- Basic understanding of Enzu concepts (agents, tools, synergies)

## Step 1: Define Assistant Capabilities

First, define your assistant's capabilities clearly. This helps users understand what the assistant can do and helps the AI provide appropriate responses:

```go
const aiCapabilities = `
I am an AI assistant with the following capabilities:

1. Research and Fact-Checking:
   - Conduct web searches on various topics
   - Verify and validate information
   - Provide summaries of researched topics

2. Data Analysis and Insights:
   - Analyze data sets
   - Identify patterns and trends
   - Generate insights and conclusions

3. Creative Content Generation:
   - Generate original content
   - Edit and refine content
   - Assist with brainstorming

[...]
`
```

## Step 2: Initialize Core Components

Set up the LLM and logger with appropriate configuration:

```go
// Create LLM instance with retry logic
llm, err := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-4o-mini"),
    gollm.SetAPIKey(openAIKey),
    gollm.SetMaxTokens(300),
    gollm.SetMaxRetries(3),
    gollm.SetRetryDelay(time.Second*2),
)

// Create logger for monitoring
logger := enzu.NewLogger(enzu.LogLevelInfo)
```

## Step 3: Register Built-in Tools

Register any built-in tools that your assistant will need:

```go
// Configure ExaSearch for web research
exaSearchOptions := tools.ExaSearchOptions{
    NumResults: 3,
    Type:       "neural",
    Contents: tools.Contents{
        Text: true,
    },
    UseAutoprompt: true,
}
tools.ExaSearch("", exaSearchOptions, "ResearchTool")
```

## Step 4: Create Specialized Synergies

Create separate synergies for different types of tasks:

```go
func createResearchSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
    // Create specialized agents
    researcher := enzu.NewAgent(
        "Web Researcher",
        "Perform web searches and summarize information",
        llm,
        enzu.WithToolLists("ResearchTool"),
    )
    factChecker := enzu.NewAgent(
        "Fact Checker",
        "Verify information and cross-reference sources",
        llm,
    )

    // Create research tasks
    tasks := []*enzu.Task{
        enzu.NewTask("Conduct research", researcher),
        enzu.NewTask("Verify findings", factChecker),
    }

    // Create and return synergy
    return enzu.NewSynergy(
        "Research Operations",
        llm,
        enzu.WithAgents(researcher, factChecker),
        enzu.WithTasks(tasks...),
        enzu.WithLogger(logger),
    )
}
```

## Step 5: Set Up the SynergyManager

Create and configure the SynergyManager to orchestrate all synergies:

```go
// Create manager
manager := enzu.NewSynergyManager(
    "Self-Aware Interactive AI Assistant",
    llm,
    logger,
)

// Add specialized synergies
manager.AddSynergy(createResearchSynergy(llm, logger))
manager.AddSynergy(createAnalysisSynergy(llm, logger))
manager.AddSynergy(createCreativeSynergy(llm, logger))
```

## Step 6: Implement Conversation Handling

Create functions to handle user input and maintain conversation history:

```go
func handleUserInput(manager *enzu.SynergyManager, input string, history []string) string {
    // Check for capability questions
    if isAskingAboutCapabilities(input) {
        return explainCapabilities()
    }

    // Create context-aware prompt
    historyContext := strings.Join(history, "\n")
    prompt := fmt.Sprintf(`
AI Capabilities:
%s

Conversation history:
%s

Current user input: %s
`, aiCapabilities, historyContext, input)

    // Execute synergies with context
    results, err := manager.ExecuteSynergies(context.Background(), prompt)
    if err != nil {
        return fmt.Sprintf("Error: %v", err)
    }

    return results["synthesis"].(string)
}
```

## Step 7: Create the Interactive Loop

Implement the main interaction loop:

```go
func main() {
    // ... initialization code ...

    reader := bufio.NewReader(os.Stdin)
    history := []string{}

    for {
        fmt.Print("\nYou: ")
        input, _ := reader.ReadString('\n')
        input = strings.TrimSpace(input)

        if input == "exit" {
            break
        }

        // Get response
        response := handleUserInput(manager, input, history)
        fmt.Printf("\nAI Assistant: %s\n", response)

        // Update history
        history = append(history,
            fmt.Sprintf("User: %s", input),
            fmt.Sprintf("Assistant: %s", response),
        )

        // Keep history manageable
        if len(history) > 10 {
            history = history[len(history)-10:]
        }
    }
}
```

## How It Works

1. **Initialization**
   - The assistant starts by setting up the LLM, logger, and tools
   - Specialized synergies are created for different types of tasks
   - The SynergyManager orchestrates all synergies

2. **User Interaction**
   - The assistant maintains a conversation history
   - Each user input is processed with historical context
   - Special queries (like capability questions) are handled directly

3. **Task Processing**
   - The manager routes tasks to appropriate synergies
   - Multiple agents collaborate within each synergy
   - Results are synthesized into a coherent response

4. **Context Management**
   - Conversation history provides context for responses
   - History is limited to recent interactions
   - AI capabilities are included in each prompt

## Best Practices

1. **Synergy Design**
   - Create focused synergies for specific capabilities
   - Use appropriate tools for each synergy
   - Keep agent responsibilities clear and distinct

2. **Error Handling**
   - Implement retry logic for LLM calls
   - Provide helpful error messages
   - Log errors for debugging

3. **User Experience**
   - Maintain clear conversation flow
   - Provide helpful responses
   - Keep response times reasonable

4. **Resource Management**
   - Limit conversation history size
   - Configure appropriate token limits
   - Monitor API usage

## Running the Example

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

2. Run the example:
   ```bash
   go run examples/7_manager_mode_example.go
   ```

3. Interact with the assistant:
   ```
   Welcome to the Self-Aware Interactive AI Assistant!
   You can ask questions, request tasks, or inquire about my capabilities.

   You: What can you do?
   AI Assistant: [Explains capabilities]

   You: Can you research recent AI developments?
   AI Assistant: [Provides researched information]
   ```

## Next Steps

1. Add more specialized synergies for new capabilities
2. Implement additional tools for specific tasks
3. Enhance the conversation history management
4. Add user authentication and session management
5. Implement persistent storage for conversation history

For the complete example code, see [manager_mode_example.go](../../examples/7_manager_mode_example.go).
