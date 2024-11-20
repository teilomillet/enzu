// Package enzu implements task execution and management within the framework's
// agent-based architecture. Tasks represent discrete units of work that agents
// execute as part of a synergy's workflow. The task system provides:
//   - Contextual task execution with access to previous results
//   - Dynamic tool discovery and execution
//   - Structured agent-LLM interaction
//   - Flexible execution patterns through the TaskExecutor interface
package enzu

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/teilomillet/gollm"
)

// Task represents a discrete unit of work within a Synergy's workflow. It encapsulates
// both the work to be done and the context needed to execute it properly.
//
// Tasks are the fundamental execution units in the Enzu framework:
//  - Assigned to specific agents for execution
//  - Can access results from previous tasks
//  - Have access to agent-specific and synergy-wide tools
//  - Support both sequential and parallel execution patterns
type Task struct {
	// description defines what the task needs to accomplish
	description string
	
	// agent is the AI entity responsible for executing this task
	agent *Agent
	
	// context contains references to previous tasks whose results may be needed
	context []*Task
}

// NewTask creates a new Task instance with the specified parameters.
// It establishes the relationship between the task, its executing agent,
// and any contextual tasks whose results may be needed.
//
// Parameters:
//   - description: Clear description of what the task should accomplish
//   - agent: The agent responsible for executing this task
//   - context: Optional slice of previous tasks providing context
//
// Returns:
//   - *Task: A new task instance ready for execution
func NewTask(description string, agent *Agent, context ...*Task) *Task {
	return &Task{
		description: description,
		agent:       agent,
		context:     context,
	}
}

// Description returns the task's description string.
// This is useful for logging, debugging, and generating prompts.
//
// Returns:
//   - string: The task's description
func (t *Task) Description() string {
	return t.description
}

// TaskExecutor defines the interface for executing tasks within the framework.
// This interface allows for different execution strategies while maintaining
// a consistent execution pattern across the system.
//
// Implementations of this interface handle:
//  - Task prompt preparation
//  - LLM interaction
//  - Tool execution
//  - Result processing
type TaskExecutor interface {
	// ExecuteTask runs a single task with the provided context and returns its result.
	//
	// Parameters:
	//   - ctx: Context for cancellation and deadline control
	//   - task: The task to execute
	//   - taskContext: Results from previous task executions
	//   - logger: Logger for tracking execution progress
	//
	// Returns:
	//   - string: The task execution result
	//   - error: Any error encountered during execution
	ExecuteTask(ctx context.Context, task *Task, taskContext map[string]string, logger *Logger) (string, error)
}

// DefaultTaskExecutor provides the standard implementation of TaskExecutor.
// It implements a straightforward execution flow that includes:
//  1. Prompt preparation with task and context
//  2. LLM interaction for decision making
//  3. Tool execution when requested
//  4. Result collection and return
type DefaultTaskExecutor struct{}

// ExecuteTask implements the TaskExecutor interface with the default execution strategy.
// It manages the complete lifecycle of a task execution, from prompt preparation
// to result collection.
//
// The execution process:
//  1. Prepares the execution context and prompt
//  2. Interacts with the agent's LLM
//  3. Handles any tool execution requests
//  4. Processes and returns the final result
//
// Parameters:
//   - ctx: Context for cancellation and deadline control
//   - task: The task to execute
//   - taskContext: Map of previous task results
//   - logger: Logger for execution tracking
//
// Returns:
//   - string: The task's execution result
//   - error: Any error encountered during execution
func (e *DefaultTaskExecutor) ExecuteTask(ctx context.Context, task *Task, taskContext map[string]string, logger *Logger) (string, error) {
	logger.Info("Task", "Executing: %s", task.description)
	prompt := preparePrompt(task, taskContext)
	logger.Debug("Prompt", "Prepared prompt for task '%s':\n%s", task.description, prompt)

	response, err := executeWithLLM(ctx, task.agent, prompt, logger)
	if err != nil {
		logger.Error("Task", "Error executing task '%s': %v", task.description, err)
		return "", err
	}

	logger.Info("Task", "Completed: %s", task.description)
	logger.Debug("Result", "Task '%s' result:\n%s", task.description, response)
	return response, nil
}

// preparePrompt constructs the prompt for LLM interaction based on the task's context.
// It builds a comprehensive prompt that includes:
//  - Agent identity and role
//  - Task description
//  - Results from contextual tasks
//  - Available tools and their usage instructions
//
// Parameters:
//   - task: The task being executed
//   - taskContext: Map of previous task results
//
// Returns:
//   - string: The formatted prompt ready for LLM interaction
func preparePrompt(task *Task, taskContext map[string]string) string {
	prompt := fmt.Sprintf("You are %s. Your role is %s.\n\n", task.agent.name, task.agent.role)
	prompt += fmt.Sprintf("Task: %s\n\n", task.description)

	if len(task.context) > 0 {
		prompt += "Context from previous tasks:\n"
		for _, contextTask := range task.context {
			if result, ok := taskContext[contextTask.description]; ok {
				prompt += fmt.Sprintf("- %s: %s\n", contextTask.description, result)
			}
		}
		prompt += "\n"
	}

	prompt += "Available tools:\n"
	toolLists := task.agent.toolLists
	if task.agent.inheritSynergyTools {
		toolLists = append(toolLists, task.agent.synergy.toolLists...)
	}
	for _, listName := range toolLists {
		if list, exists := task.agent.synergy.toolRegistry.GetToolList(listName); exists {
			for _, tool := range list.Tools {
				prompt += fmt.Sprintf("- %s: %s\n", tool.Name, tool.Description)
			}
		}
	}
	prompt += "\nTo use a tool, respond with a JSON object in this format: {\"tool\": \"ToolName\", \"args\": [arg1, arg2, ...]}\n\n"

	prompt += "Please complete this task based on the information provided. If you need to use a tool, respond with the appropriate JSON object."
	return prompt
}

// executeWithLLM manages the interaction between the agent's LLM and available tools.
// It implements a conversation loop that allows the LLM to:
//  1. Analyze the task and context
//  2. Decide whether to use tools
//  3. Execute tools and process their results
//  4. Generate the final response
//
// Parameters:
//   - ctx: Context for cancellation and deadline control
//   - agent: The agent executing the task
//   - prompt: The prepared prompt for LLM interaction
//   - logger: Logger for tracking the execution
//
// Returns:
//   - string: The final execution result
//   - error: Any error encountered during execution
func executeWithLLM(ctx context.Context, agent *Agent, prompt string, logger *Logger) (string, error) {
	for {
		response, err := agent.llm.Generate(ctx, gollm.NewPrompt(prompt))
		if err != nil {
			return "", fmt.Errorf("LLM generation error: %w", err)
		}

		// Check if the response is a tool execution request
		var toolRequest struct {
			Tool string        `json:"tool"`
			Args []interface{} `json:"args"`
		}

		if err := json.Unmarshal([]byte(response), &toolRequest); err == nil && toolRequest.Tool != "" {
			// Execute the tool
			tool, exists := agent.synergy.toolRegistry.GetTool(toolRequest.Tool)
			if !exists {
				logger.Warn("Tool", "Unknown tool requested: %s", toolRequest.Tool)
				prompt += fmt.Sprintf("\nThe tool '%s' is not available. Please try again with an available tool or complete the task without a tool.", toolRequest.Tool)
				continue
			}

			result, err := tool.Execute(toolRequest.Args...)
			if err != nil {
				logger.Error("Tool", "Error executing tool '%s': %v", toolRequest.Tool, err)
				prompt += fmt.Sprintf("\nThere was an error executing the tool '%s': %v. Please try again or complete the task without this tool.", toolRequest.Tool, err)
				continue
			}

			// Add tool result to the prompt and continue the conversation
			prompt += fmt.Sprintf("\nTool '%s' executed successfully. Result: %v\nPlease complete the task based on this result.", toolRequest.Tool, result)
		} else {
			// If it's not a tool request, return the response
			return response, nil
		}
	}
}
