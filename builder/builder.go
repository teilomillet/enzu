// Package builder provides functionality for constructing AI agents and synergies from
// configuration files. It serves as a factory for creating complex AI system components
// with specific behaviors and capabilities defined through configuration.
package builder

import (
	"fmt"

	"github.com/teilomillet/enzu"
	"github.com/teilomillet/gollm"
)

// Builder handles the creation of agents and synergies from configurations.
// It acts as a factory for constructing complex AI system components based on
// provided configurations, managing the instantiation of LLMs, agents, and their
// associated tools and behaviors.
type Builder struct {
	config *SynergyConfig
}

// NewBuilder creates a new Builder instance with the provided synergy configuration.
// It initializes a builder that can construct agents and synergies based on the
// configuration specifications.
//
// Parameters:
//   - config: A pointer to a SynergyConfig containing the complete system configuration
//
// Returns:
//   - *Builder: A new builder instance ready to construct AI system components
func NewBuilder(config *SynergyConfig) *Builder {
	return &Builder{
		config: config,
	}
}

// createLLM creates an LLM (Large Language Model) instance from the provided configuration.
// It handles the setup of different LLM types (e.g., GPT-4, GPT-3) with their specific
// parameters and API configurations.
//
// Parameters:
//   - config: LLMConfig containing the model type and parameters
//
// Returns:
//   - gollm.LLM: The configured LLM instance
//   - error: Any error encountered during LLM creation
//
// Supported LLM types:
//   - "gpt4": OpenAI's GPT-4 model
//   - "gpt3": OpenAI's GPT-3.5 Turbo model
func (b *Builder) createLLM(config LLMConfig) (gollm.LLM, error) {
	// Convert the generic parameters map to specific gollm options
	var opts []gollm.ConfigOption

	// Set the provider and model based on type
	switch config.Type {
	case "gpt4":
		opts = append(opts,
			gollm.SetProvider("openai"),
			gollm.SetModel("gpt-4o-mini"),
		)
	case "gpt3":
		opts = append(opts,
			gollm.SetProvider("openai"),
			gollm.SetModel("gpt-3.5-turbo"),
		)
	default:
		return nil, fmt.Errorf("unsupported LLM type: %s", config.Type)
	}

	// Apply any additional parameters from config
	if maxTokens, ok := config.Parameters["max_tokens"].(float64); ok {
		opts = append(opts, gollm.SetMaxTokens(int(maxTokens)))
	}
	if apiKey, ok := config.Parameters["api_key"].(string); ok {
		opts = append(opts, gollm.SetAPIKey(apiKey))
	}

	return gollm.NewLLM(opts...)
}

// BuildAgent creates a new Agent from the provided configuration. It sets up the agent
// with specified tools, behaviors, and LLM capabilities.
//
// Parameters:
//   - config: AgentConfig containing the agent's name, role, LLM configuration, and behavior settings
//
// Returns:
//   - *enzu.Agent: The constructed agent instance
//   - error: Any error encountered during agent creation
func (b *Builder) BuildAgent(config AgentConfig) (*enzu.Agent, error) {
	llm, err := b.createLLM(config.LLMConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM for agent %s: %v", config.Name, err)
	}

	opts := []enzu.AgentOption{
		enzu.WithToolLists(config.ToolLists...),
		enzu.WithInheritSynergyTools(config.InheritSynergyTools),
		enzu.WithParallelExecution(config.Parallel),
	}

	return enzu.NewAgent(config.Name, config.Role, llm, opts...), nil
}

// BuildSynergy creates a new Synergy from the builder's configuration. A Synergy
// represents a complete AI system with multiple agents working together towards
// a common objective.
//
// The method:
//  1. Creates the main LLM for the synergy
//  2. Builds all configured agents
//  3. Sets up the synergy with tools and agents
//
// Returns:
//   - *enzu.Synergy: The constructed synergy instance
//   - error: Any error encountered during synergy creation
func (b *Builder) BuildSynergy() (*enzu.Synergy, error) {
	llm, err := b.createLLM(b.config.LLM)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM for synergy: %v", err)
	}

	agents := make([]*enzu.Agent, 0, len(b.config.Agents))
	for _, agentConfig := range b.config.Agents {
		agent, err := b.BuildAgent(agentConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to build agent %s: %v", agentConfig.Name, err)
		}
		agents = append(agents, agent)
	}

	opts := []enzu.SynergyOption{
		enzu.WithTools(b.config.ToolLists...),
		enzu.WithAgents(agents...),
	}

	return enzu.NewSynergy(b.config.Objective, llm, opts...), nil
}
