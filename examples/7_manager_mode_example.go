package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/teilomillet/enzu"
	"github.com/teilomillet/enzu/tools"
	"github.com/teilomillet/gollm"
)

// Define the AI's capabilities
const aiCapabilities = `
I am an AI assistant with the following capabilities:

1. Research and Fact-Checking:
   - Conduct web searches on various topics
   - Verify and validate information from multiple sources
   - Provide summaries of researched topics

2. Data Analysis and Insights:
   - Analyze provided data sets or information
   - Identify patterns and trends in data
   - Generate insights and conclusions based on analysis
   - Provide data visualization suggestions

3. Creative Content Generation:
   - Generate original content based on given prompts or themes
   - Edit and refine existing content
   - Assist with brainstorming ideas for various creative projects

4. General Knowledge and Q&A:
   - Answer questions on a wide range of topics
   - Explain complex concepts in simple terms
   - Provide definitions and explanations

5. Task Planning and Problem-Solving:
   - Break down complex tasks into manageable steps
   - Suggest approaches to solve various problems
   - Help with decision-making by providing pros and cons

To make the best use of my capabilities, please provide clear and specific prompts or questions. If you need clarification on any of my functions, feel free to ask!
`

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
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(time.Second*2),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to initialize LLM: %v", err)
	}

	// Create a logger for Enzu
	logger := enzu.NewLogger(enzu.LogLevelInfo)

	// Register the ExaSearch tool
	exaSearchOptions := tools.ExaSearchOptions{
		NumResults: 3,
		Type:       "neural",
		Contents: tools.Contents{
			Text: true,
		},
		UseAutoprompt:      true,
		StartPublishedDate: "2023-01-01T00:00:00.000Z",
	}
	tools.ExaSearch("", exaSearchOptions, "ResearchTool")

	// Create SynergyManager
	manager := enzu.NewSynergyManager("Self-Aware Interactive AI Assistant", llm, logger)

	// Create Synergies
	researchSynergy := createResearchSynergy(llm, logger)
	analysisSynergy := createAnalysisSynergy(llm, logger)
	creativeSynergy := createCreativeSynergy(llm, logger)

	manager.AddSynergy(researchSynergy)
	manager.AddSynergy(analysisSynergy)
	manager.AddSynergy(creativeSynergy)

	// Start interactive chat
	fmt.Println("Welcome to the Self-Aware Interactive AI Assistant!")
	fmt.Println("You can ask questions, request tasks, or inquire about my capabilities. Type 'exit' to quit.")

	reader := bufio.NewReader(os.Stdin)
	conversationHistory := []string{}

	for {
		fmt.Print("\nYou: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" {
			fmt.Println("Goodbye!")
			break
		}

		response := handleUserInput(manager, input, conversationHistory)
		fmt.Printf("\nAI Assistant: %s\n", response)

		// Update conversation history
		conversationHistory = append(conversationHistory, fmt.Sprintf("User: %s", input))
		conversationHistory = append(conversationHistory, fmt.Sprintf("Assistant: %s", response))

		// Limit history to last 5 interactions (10 messages)
		if len(conversationHistory) > 10 {
			conversationHistory = conversationHistory[len(conversationHistory)-10:]
		}
	}
}

func handleUserInput(manager *enzu.SynergyManager, input string, history []string) string {
	// Check if the user is asking about capabilities
	if isAskingAboutCapabilities(input) {
		return explainCapabilities()
	}

	ctx := context.Background()

	// Create a prompt that includes conversation history and AI capabilities
	historyContext := strings.Join(history, "\n")
	prompt := fmt.Sprintf(`
AI Capabilities:
%s

Conversation history:
%s

Current user input: %s

Please provide a response based on your capabilities, the conversation history, and the current input.
`, aiCapabilities, historyContext, input)

	results, err := manager.ExecuteSynergies(ctx, prompt)
	if err != nil {
		return fmt.Sprintf("I apologize, but I encountered an error while processing your request: %v", err)
	}

	return results["synthesis"].(string)
}

func isAskingAboutCapabilities(input string) bool {
	lowercaseInput := strings.ToLower(input)
	capabilityKeywords := []string{"what can you do", "your capabilities", "what are you capable of", "your functions", "your abilities"}

	for _, keyword := range capabilityKeywords {
		if strings.Contains(lowercaseInput, keyword) {
			return true
		}
	}
	return false
}

func explainCapabilities() string {
	return `Certainly! I'd be happy to explain my capabilities. As an AI assistant, I'm designed to help with a variety of tasks:

1. Research and Fact-Checking:
   I can conduct web searches on various topics, verify information from multiple sources, and provide summaries of researched topics.

2. Data Analysis and Insights:
   I can analyze provided data sets, identify patterns and trends, generate insights and conclusions, and even suggest data visualizations.

3. Creative Content Generation:
   I can generate original content based on prompts or themes, edit and refine existing content, and assist with brainstorming ideas for creative projects.

4. General Knowledge and Q&A:
   I can answer questions on a wide range of topics, explain complex concepts in simple terms, and provide definitions and explanations.

5. Task Planning and Problem-Solving:
   I can help break down complex tasks into manageable steps, suggest approaches to solve various problems, and assist with decision-making by providing pros and cons.

To make the best use of my capabilities, it's helpful if you provide clear and specific prompts or questions. Feel free to ask for clarification or more details about any of these functions!

What kind of task can I help you with today?`
}

func createResearchSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
	agents := []*enzu.Agent{
		enzu.NewAgent("Web Researcher", "Perform web searches and summarize information", llm, enzu.WithToolLists("ResearchTool")),
		enzu.NewAgent("Fact Checker", "Verify information and cross-reference sources", llm),
	}

	tasks := []*enzu.Task{
		enzu.NewTask("Conduct web research on the given topic", agents[0]),
		enzu.NewTask("Verify and validate the researched information", agents[1]),
	}

	return enzu.NewSynergy(
		"Research and Fact-Checking",
		llm,
		enzu.WithAgents(agents...),
		enzu.WithTasks(tasks...),
		enzu.WithLogger(logger),
	)
}

func createAnalysisSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
	agents := []*enzu.Agent{
		enzu.NewAgent("Data Analyst", "Analyze data and identify patterns", llm),
		enzu.NewAgent("Insight Generator", "Generate insights from analyzed data", llm),
	}

	tasks := []*enzu.Task{
		enzu.NewTask("Analyze the provided information or data", agents[0]),
		enzu.NewTask("Generate insights and conclusions from the analysis", agents[1]),
	}

	return enzu.NewSynergy(
		"Data Analysis and Insights",
		llm,
		enzu.WithAgents(agents...),
		enzu.WithTasks(tasks...),
		enzu.WithLogger(logger),
	)
}

func createCreativeSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
	agents := []*enzu.Agent{
		enzu.NewAgent("Creative Writer", "Generate creative content and ideas", llm),
		enzu.NewAgent("Editor", "Refine and polish creative content", llm),
	}

	tasks := []*enzu.Task{
		enzu.NewTask("Generate creative content based on the given prompt", agents[0]),
		enzu.NewTask("Edit and refine the generated creative content", agents[1]),
	}

	return enzu.NewSynergy(
		"Creative Content Generation",
		llm,
		enzu.WithAgents(agents...),
		enzu.WithTasks(tasks...),
		enzu.WithLogger(logger),
	)
}
