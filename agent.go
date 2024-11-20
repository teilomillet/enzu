// Package enzu provides the core functionality for building and managing AI agents and synergies.
// It implements a flexible framework for creating AI agents with specific roles, capabilities,
// and behaviors, allowing them to work independently or collaboratively within a synergy.
package enzu

import (
	"github.com/teilomillet/gollm"
)

// Agent represents an AI agent with a specific role and capabilities within the Enzu framework.
// Each agent has its own identity, language model, tool access, and execution preferences.
// Agents can work independently or as part of a larger synergy, sharing tools and coordinating
// actions with other agents.
type Agent struct {
	// name is the unique identifier for this agent
	name string
	
	// role defines the agent's purpose and expected behavior
	role string
	
	// llm is the language model powering this agent's intelligence
	llm gollm.LLM
	
	// toolLists contains the names of tool collections this agent can access
	toolLists []string
	
	// inheritSynergyTools determines if the agent inherits tools from its parent synergy
	inheritSynergyTools bool
	
	// synergy is a reference to the parent synergy this agent belongs to
	synergy *Synergy
	
	// parallel indicates whether this agent can execute tasks concurrently
	parallel bool
}

// AgentOption is a function type for configuring an Agent using the functional options pattern.
// It allows for flexible and extensible agent configuration without breaking existing code
// when new options are added.
type AgentOption func(*Agent)

// NewAgent creates a new Agent with the specified name, role, language model, and options.
// It initializes the agent with default settings that can be overridden using option functions.
//
// Parameters:
//   - name: Unique identifier for the agent
//   - role: Description of the agent's purpose and behavior
//   - llm: Language model instance that powers the agent
//   - opts: Variable number of AgentOption functions for additional configuration
//
// Returns:
//   - *Agent: A new agent instance configured with the specified options
//
// By default, agents are configured to:
//   - Inherit tools from their parent synergy
//   - Execute tasks sequentially (non-parallel)
func NewAgent(name, role string, llm gollm.LLM, opts ...AgentOption) *Agent {
	a := &Agent{
		name:                name,
		role:                role,
		llm:                 llm,
		inheritSynergyTools: true,  // Default to inheriting Synergy tools
		parallel:            false, // Default to sequential execution
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// WithToolLists specifies which ToolLists the Agent has access to.
// This option allows you to grant the agent access to specific sets of tools
// that it can use to accomplish its tasks.
//
// Parameters:
//   - lists: Variable number of tool list names to assign to the agent
//
// Returns:
//   - AgentOption: A function that configures the agent's tool lists when applied
func WithToolLists(lists ...string) AgentOption {
	return func(a *Agent) {
		a.toolLists = lists
	}
}

// WithInheritSynergyTools controls whether the Agent inherits tool lists from its Synergy.
// When enabled, the agent can access tools available to its parent synergy in addition
// to its own tool lists.
//
// Parameters:
//   - inherit: If true, the agent will inherit tools from its parent synergy
//
// Returns:
//   - AgentOption: A function that configures the agent's tool inheritance when applied
func WithInheritSynergyTools(inherit bool) AgentOption {
	return func(a *Agent) {
		a.inheritSynergyTools = inherit
	}
}

// WithParallelExecution sets whether tasks assigned to this agent should be executed in parallel.
// When enabled, the agent can process multiple tasks concurrently, potentially improving
// performance for independent operations.
//
// Parameters:
//   - parallel: If true, enables parallel task execution for this agent
//
// Returns:
//   - AgentOption: A function that configures the agent's execution mode when applied
func WithParallelExecution(parallel bool) AgentOption {
	return func(a *Agent) {
		a.parallel = parallel
	}
}
