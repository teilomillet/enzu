package enzu

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/teilomillet/gollm"
)

// Task represents a specific task to be performed
type Task struct {
	description string
	agent       *Agent
	context     []*Task
}

// NewTask creates a new Task
func NewTask(description string, agent *Agent, context ...*Task) *Task {
	return &Task{
		description: description,
		agent:       agent,
		context:     context,
	}
}

// Description returns the task description
func (t *Task) Description() string {
	return t.description
}

// TaskExecutor defines the interface for executing tasks
type TaskExecutor interface {
	ExecuteTask(ctx context.Context, task *Task, taskContext map[string]string, logger *Logger) (string, error)
}

// DefaultTaskExecutor is the default implementation of TaskExecutor
type DefaultTaskExecutor struct{}

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
