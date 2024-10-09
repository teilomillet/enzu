package enzu

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm"
)

// Synergy represents a collaboration of AI agents working towards a common goal
type Synergy struct {
	objective    string
	llm          gollm.LLM
	agents       []*Agent
	tasks        []*Task
	executor     TaskExecutor
	logger       *Logger
	toolRegistry *ToolRegistry
	toolLists    []string
	parallel     bool // New field to control parallel execution
}

// SynergyOption is a function type for configuring a Synergy
type SynergyOption func(*Synergy)

// NewSynergy creates a new Synergy instance with the given options
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

// WithLogger sets the logger for the Synergy
func WithLogger(logger *Logger) SynergyOption {
	return func(s *Synergy) {
		s.logger = logger
	}
}

// WithToolRegistry sets a custom ToolRegistry for the Synergy
func WithToolRegistry(registry *ToolRegistry) SynergyOption {
	return func(s *Synergy) {
		s.toolRegistry = registry
	}
}

// WithTools specifies which ToolLists the Synergy has access to
func WithTools(lists ...string) SynergyOption {
	return func(s *Synergy) {
		s.toolLists = lists
	}
}

// WithAgents adds agents to the Synergy
func WithAgents(agents ...*Agent) SynergyOption {
	return func(s *Synergy) {
		s.agents = append(s.agents, agents...)
		for _, agent := range agents {
			agent.synergy = s
		}
	}
}

// WithTasks adds tasks to the Synergy
func WithTasks(tasks ...*Task) SynergyOption {
	return func(s *Synergy) {
		s.tasks = append(s.tasks, tasks...)
	}
}

// GetAgents returns the list of agents in the Synergy
func (s *Synergy) GetAgents() []*Agent {
	return s.agents
}

// GetTasks returns the list of tasks in the Synergy
func (s *Synergy) GetTasks() []*Task {
	return s.tasks
}

// SetTasks sets the tasks for the Synergy
func (s *Synergy) SetTasks(tasks []*Task) {
	s.tasks = tasks
}

// AddTask adds a new task to the Synergy
func (s *Synergy) AddTask(task *Task) {
	s.tasks = append(s.tasks, task)
}

// Execute runs the Synergy, executing all tasks
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

// executeTaskInParallel executes a task in parallel using goroutines
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

// taskResult holds the result of a task execution
type taskResult struct {
	task   *Task
	result string
}
