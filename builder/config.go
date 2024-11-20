// Package builder provides configuration structures and loading functionality for the Enzu framework.
// This file specifically handles the definition and loading of configuration structures
// for agents, LLMs, and synergies from YAML or JSON files.
package builder

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// AgentConfig represents the configuration for an AI agent within the Enzu framework.
// It defines the agent's identity, role, language model configuration, and operational parameters.
type AgentConfig struct {
	// Name is the unique identifier for the agent
	Name                string    `yaml:"name" json:"name"`
	
	// Role defines the agent's purpose and behavior within the system
	Role                string    `yaml:"role" json:"role"`
	
	// LLMConfig specifies the language model configuration for this agent
	LLMConfig           LLMConfig `yaml:"llm" json:"llm"`
	
	// ToolLists contains the names of tool collections available to this agent
	ToolLists           []string  `yaml:"tool_lists,omitempty" json:"tool_lists,omitempty"`
	
	// InheritSynergyTools determines if the agent should inherit tools from its parent synergy
	InheritSynergyTools bool      `yaml:"inherit_synergy_tools" json:"inherit_synergy_tools"`
	
	// Parallel indicates whether the agent can execute tasks in parallel
	Parallel            bool      `yaml:"parallel" json:"parallel"`
}

// LLMConfig represents the configuration for a Large Language Model (LLM).
// It specifies the model type and its operational parameters.
type LLMConfig struct {
	// Type specifies the LLM model type (e.g., "gpt4", "gpt3")
	Type       string                 `yaml:"type" json:"type"`
	
	// Parameters contains model-specific configuration options
	Parameters map[string]interface{} `yaml:"parameters" json:"parameters"`
}

// SynergyConfig represents the configuration for an entire AI system (Synergy).
// It defines the system's objective, main LLM configuration, and constituent agents.
type SynergyConfig struct {
	// Objective defines the main goal or purpose of the synergy
	Objective string        `yaml:"objective" json:"objective"`
	
	// LLM specifies the primary language model configuration for the synergy
	LLM       LLMConfig     `yaml:"llm" json:"llm"`
	
	// Agents contains the configurations for all agents in the synergy
	Agents    []AgentConfig `yaml:"agents" json:"agents"`
	
	// ToolLists specifies the names of tool collections available to the synergy
	ToolLists []string      `yaml:"tool_lists,omitempty" json:"tool_lists,omitempty"`
	
	// Parallel indicates whether the synergy can execute tasks in parallel
	Parallel  bool          `yaml:"parallel" json:"parallel"`
}

// LoadConfig loads a configuration file (YAML or JSON) and returns a SynergyConfig.
// It supports both YAML (.yaml, .yml) and JSON (.json) file formats.
//
// Parameters:
//   - configPath: The path to the configuration file
//
// Returns:
//   - *SynergyConfig: A pointer to the loaded configuration
//   - error: Any error encountered during loading or parsing
//
// The function will return an error if:
//   - The file cannot be read
//   - The file format is not supported (must be .yaml, .yml, or .json)
//   - The file contents cannot be parsed into the configuration structure
func LoadConfig(configPath string) (*SynergyConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %v", err)
	}

	var config SynergyConfig
	ext := filepath.Ext(configPath)

	switch ext {
	case ".yaml", ".yml":
		err = yaml.Unmarshal(data, &config)
	case ".json":
		err = json.Unmarshal(data, &config)
	default:
		return nil, fmt.Errorf("unsupported config file format: %s", ext)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to parse config file: %v", err)
	}

	return &config, nil
}
