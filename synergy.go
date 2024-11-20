// Package enzu implements a multi-agent AI orchestration system where Synergy acts as
// the middle layer, coordinating multiple AI agents working towards a common objective.
// Synergy sits between the high-level SynergyManager and individual Agents, providing:
//   - Task orchestration and execution management
//   - Agent collaboration and coordination
//   - Tool and resource sharing
//   - Contextual state management across tasks
package enzu

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm"
)

// Synergy represents a collaborative AI workflow where multiple agents work together
// towards a common objective. It serves as the primary orchestration unit in the
// Enzu framework, managing task execution, agent coordination, and resource sharing.
//
// In the framework's hierarchy:
//  - Above: SynergyManager coordinates multiple Synergies
//  - Current: Synergy orchestrates multiple Agents
//  - Below: Agents execute individual tasks
//
// Key responsibilities:
//  1. Task Management: Coordinates task execution across agents
//  2. Resource Sharing: Manages shared tools and context
//  3. Execution Control: Handles sequential/parallel execution
//  4. State Management: Maintains task context and results
type Synergy struct {
	// objective defines the common goal for this collaborative workflow
	objective string
	
	// llm is the language model used for task coordination
	llm gollm.LLM
	
	// agents are the AI entities working together in this synergy
	agents []*Agent
	
	// tasks is the sequence of operations to be executed
	tasks []*Task
	
	// executor handles the actual execution of tasks
	executor TaskExecutor
	
	// logger provides logging capabilities for this synergy
	logger *Logger
	
	// toolRegistry manages available tools for all agents
	toolRegistry *ToolRegistry
	
	// toolLists specifies which tool collections are available
	toolLists []string
	
	// parallel indicates whether tasks can be executed concurrently
	parallel bool
}

// SynergyOption is a function type for configuring a Synergy using the functional
// options pattern. This allows for flexible and extensible configuration without
// breaking existing code when new features are added.
type SynergyOption func(*Synergy)

// NewSynergy creates a new Synergy instance that will coordinate multiple agents
// working towards a common objective. It initializes the synergy with default
// settings that can be customized through options.
//
// Parameters:
//   - objective: The common goal this synergy works towards
//   - llm: Language model for task coordination
//   - opts: Configuration options for customizing behavior
//
// Returns:
//   - *Synergy: A new synergy instance ready to coordinate agents
//
// By default, synergies are configured with:
//   - Sequential task execution
//   - Info-level logging
//   - Default tool registry
func NewSynergy(objective string, llm gollm.LLM, opts ...SynergyOption) *Synergy {
	s := &Synergy{
		objective:    objective,
		llm:          llm,
		executor:     &DefaultTaskExecutor{},
		logger:       NewLogger(LogLevelInfo),
		toolRegistry: defaultRegistry,
		parallel:     false, // Default to sequential execution
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// WithLogger configures the logging system for this synergy.
// The logger tracks execution progress, debug information, and errors
// across all tasks and agents within this synergy.
//
// Parameters:
//   - logger: The logger instance to use
func WithLogger(logger *Logger) SynergyOption {
	return func(s *Synergy) {
		s.logger = logger
	}
}

// WithToolRegistry configures a custom tool registry for this synergy.
// The tool registry manages which tools are available to agents and
// how they can be accessed during task execution.
//
// Parameters:
//   - registry: Custom tool registry to use
func WithToolRegistry(registry *ToolRegistry) SynergyOption {
	return func(s *Synergy) {
		s.toolRegistry = registry
	}
}

// WithTools specifies which tool collections are available to this synergy.
// These tools will be available to all agents unless specifically restricted.
//
// Parameters:
//   - lists: Names of tool collections to make available
func WithTools(lists ...string) SynergyOption {
	return func(s *Synergy) {
		s.toolLists = lists
	}
}

// WithAgents adds agents to this synergy and establishes the bidirectional
// relationship between agents and their synergy. Each agent becomes part
// of the collaborative workflow and gains access to shared resources.
//
// Parameters:
//   - agents: The agents to add to this synergy
func WithAgents(agents ...*Agent) SynergyOption {
	return func(s *Synergy) {
		s.agents = append(s.agents, agents...)
		for _, agent := range agents {
			agent.synergy = s
		}
	}
}

// WithTasks configures the sequence of tasks this synergy will execute.
// Tasks represent the concrete steps needed to achieve the synergy's objective.
//
// Parameters:
//   - tasks: The tasks to be executed
func WithTasks(tasks ...*Task) SynergyOption {
	return func(s *Synergy) {
		s.tasks = append(s.tasks, tasks...)
	}
}

// GetAgents returns the list of agents participating in this synergy.
// This is useful for inspecting the current state of agent collaboration.
//
// Returns:
//   - []*Agent: Slice of all agents in this synergy
func (s *Synergy) GetAgents() []*Agent {
	return s.agents
}

// GetTasks returns the current sequence of tasks in this synergy.
// This allows inspection of the workflow's structure and progress.
//
// Returns:
//   - []*Task: Slice of all tasks in this synergy
func (s *Synergy) GetTasks() []*Task {
	return s.tasks
}

// SetTasks updates the entire task sequence for this synergy.
// This is useful when dynamically generating or modifying workflows.
//
// Parameters:
//   - tasks: The new set of tasks to execute
func (s *Synergy) SetTasks(tasks []*Task) {
	s.tasks = tasks
}

// AddTask appends a new task to this synergy's workflow.
// This allows for dynamic expansion of the workflow during execution.
//
// Parameters:
//   - task: The new task to add
func (s *Synergy) AddTask(task *Task) {
	s.tasks = append(s.tasks, task)
}

// Execute runs the entire synergy workflow, coordinating all tasks across agents.
// It maintains a shared context between tasks and supports both sequential and
// parallel execution modes.
//
// The execution process:
//  1. Initializes execution context and result storage
//  2. Executes each task in sequence or parallel based on configuration
//  3. Maintains task context for inter-task communication
//  4. Collects and aggregates results from all tasks
//
// Parameters:
//   - ctx: Context for cancellation and deadline control
//
// Returns:
//   - map[string]interface{}: Results from all executed tasks
//   - error: Any error encountered during execution
func (s *Synergy) Execute(ctx context.Context) (map[string]interface{}, error) {
	s.logger.Info("Synergy", "Starting execution for objective: %s", s.objective)
	s.logger.Debug("Synergy", "Number of agents: %d, Number of tasks: %d", len(s.agents), len(s.tasks))

	results := make(map[string]interface{})
	taskContext := make(map[string]string)

	for i, task := range s.tasks {
		s.logger.Info("Synergy", "Starting task %d/%d: %s", i+1, len(s.tasks), task.description)
		s.logger.Debug("Synergy", "Executing task with agent: %s", task.agent.name)

		var response string
		var err error

		if task.agent.parallel {
			// Execute task in parallel
			response, err = s.executeTaskInParallel(ctx, task, taskContext)
			if err != nil {
				s.logger.Error("synergy", "Error executing task '%s': %v", task.description, err)
				return nil, fmt.Errorf("error executing task '%s': %w", task.description, err)
			}
		} else {
			// Execute task sequentially
			response, err = s.executor.ExecuteTask(ctx, task, taskContext, s.logger)
			if err != nil {
				s.logger.Error("synergy", "Error executing task '%s': %v", task.description, err)
				return nil, fmt.Errorf("error executing task '%s': %w", task.description, err)
			}
		}

		results[task.description] = response
		taskContext[task.description] = response

		s.logger.Info("synergy", "Completed task %d/%d: %s", i+1, len(s.tasks), task.description)
		s.logger.Debug("synergy", "Task result: %s", response)
	}

	s.logger.Info("synergy", "Execution completed successfully")
	s.logger.Debug("synergy", "Final results: %v", results)
	return results, nil
}

// executeTaskInParallel executes a task in parallel using goroutines.
// It manages the complexity of parallel execution while ensuring proper
// error handling and result collection.
//
// Parameters:
//   - ctx: Context for cancellation and deadline control
//   - task: The task to execute in parallel
//   - taskContext: Shared context from previous task executions
//
// Returns:
//   - string: The task execution result
//   - error: Any error encountered during execution
func (s *Synergy) executeTaskInParallel(ctx context.Context, task *Task, taskContext map[string]string) (string, error) {
	taskCh := make(chan *Task, 1)
	resultCh := make(chan *taskResult, 1)
	errorCh := make(chan error, 1)

	// Start worker goroutine
	go func() {
		for t := range taskCh {
			response, err := s.executor.ExecuteTask(ctx, t, taskContext, s.logger)
			if err != nil {
				errorCh <- err
			} else {
				resultCh <- &taskResult{task: t, result: response}
			}
		}
	}()

	// Send task to worker
	taskCh <- task
	close(taskCh)

	// Collect result or error
	select {
	case err := <-errorCh:
		return "", err
	case result := <-resultCh:
		return result.result, nil
	}
}

// taskResult is an internal type used to collect results from parallel task execution.
// It associates task results with their corresponding tasks for proper tracking.
type taskResult struct {
	// task is the executed task
	task *Task
	
	// result contains the task's output
	result string
}
