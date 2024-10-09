package enzu

import (
	"github.com/teilomillet/gollm"
)

// Agent represents an AI agent with a specific role
type Agent struct {
	name                string
	role                string
	llm                 gollm.LLM
	toolLists           []string
	inheritSynergyTools bool
	synergy             *Synergy
	parallel            bool // New field to control parallel execution
}

// AgentOption is a function type for configuring an Agent
type AgentOption func(*Agent)

// NewAgent creates a new Agent with the given options
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

// WithToolLists specifies which ToolLists the Agent has access to
func WithToolLists(lists ...string) AgentOption {
	return func(a *Agent) {
		a.toolLists = lists
	}
}

// WithInheritSynergyTools controls whether the Agent inherits tool lists from its Synergy
func WithInheritSynergyTools(inherit bool) AgentOption {
	return func(a *Agent) {
		a.inheritSynergyTools = inherit
	}
}

// WithParallelExecution sets whether tasks assigned to this agent should be executed in parallel
func WithParallelExecution(parallel bool) AgentOption {
	return func(a *Agent) {
		a.parallel = parallel
	}
}
