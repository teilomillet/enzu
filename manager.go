// Package enzu implements a hierarchical AI orchestration system where SynergyManager
// sits at the top level, coordinating multiple Synergies. Each Synergy, in turn,
// manages multiple Agents working together toward specific objectives.
//
// The manager layer provides high-level orchestration capabilities:
//   - Parallel execution of multiple Synergies
//   - Result synthesis across different AI workflows
//   - Centralized logging and error handling
//   - Cross-Synergy coordination and conflict resolution
package enzu

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm"
)

// SynergyManager is the top-level coordinator in the Enzu framework's hierarchy.
// It orchestrates multiple Synergies, each representing a distinct AI workflow,
// and synthesizes their results into coherent outputs.
//
// The manager serves several key purposes in the framework:
//  1. Workflow Orchestration: Coordinates multiple AI workflows running in parallel
//  2. Result Synthesis: Combines outputs from different Synergies into meaningful insights
//  3. Resource Management: Centralizes LLM usage and logging across workflows
//  4. Error Handling: Provides system-wide error management and recovery
type SynergyManager struct {
	// name identifies this manager instance
	name string
	
	// llm is the language model used for result synthesis
	llm gollm.LLM
	
	// synergies are the AI workflows being managed
	synergies []*Synergy
	
	// logger provides centralized logging for all managed Synergies
	logger *Logger
}

// NewSynergyManager creates a new SynergyManager instance that will coordinate
// multiple AI workflows through their respective Synergies.
//
// The manager uses its own LLM instance specifically for synthesizing results
// across different Synergies, ensuring that cross-workflow insights can be
// generated without interfering with individual Synergy operations.
//
// Parameters:
//   - name: Identifier for this manager instance
//   - llm: Language model used for result synthesis
//   - logger: Centralized logger for all managed workflows
//
// Returns:
//   - *SynergyManager: A new manager instance ready to coordinate AI workflows
func NewSynergyManager(name string, llm gollm.LLM, logger *Logger) *SynergyManager {
	return &SynergyManager{
		name:      name,
		llm:       llm,
		synergies: make([]*Synergy, 0),
		logger:    logger,
	}
}

// AddSynergy registers a new Synergy with the manager for coordination.
// Each Synergy represents a distinct AI workflow with its own objective,
// agents, and tools. The manager will execute and coordinate all registered
// Synergies during workflow execution.
//
// Parameters:
//   - s: The Synergy to be managed
func (sm *SynergyManager) AddSynergy(s *Synergy) {
	sm.synergies = append(sm.synergies, s)
}

// ExecuteSynergies runs all managed Synergies and synthesizes their results
// into a coherent output. This is the main entry point for executing complex
// AI workflows that require coordination across multiple objectives.
//
// The execution process:
//  1. Runs all Synergies in sequence (future: parallel execution)
//  2. Collects results from each Synergy
//  3. Synthesizes all results using the manager's LLM
//  4. Provides a unified view of the entire workflow's output
//
// Parameters:
//   - ctx: Context for cancellation and deadline control
//   - initialPrompt: The original user request or objective
//
// Returns:
//   - map[string]interface{}: Synthesized results from all Synergies
//   - error: Any error encountered during execution or synthesis
func (sm *SynergyManager) ExecuteSynergies(ctx context.Context, initialPrompt string) (map[string]interface{}, error) {
	sm.logger.Info("SynergyManager", "Starting execution of all Synergies")

	allResults := make(map[string]interface{})

	for _, synergy := range sm.synergies {
		sm.logger.Info("SynergyManager", "Executing Synergy: %s", synergy.objective)
		results, err := synergy.Execute(ctx)
		if err != nil {
			sm.logger.Error("SynergyManager", "Error executing Synergy %s: %v", synergy.objective, err)
			return nil, err
		}

		for k, v := range results {
			allResults[fmt.Sprintf("%s:%s", synergy.objective, k)] = v
		}
	}

	// Use the manager's LLM to synthesize the results
	synthesizedResults, err := sm.synthesizeResults(ctx, allResults, initialPrompt)
	if err != nil {
		sm.logger.Error("SynergyManager", "Error synthesizing results: %v", err)
		return nil, err
	}

	sm.logger.Info("SynergyManager", "All Synergies executed and results synthesized")
	return synthesizedResults, nil
}

// synthesizeResults combines outputs from multiple Synergies into a coherent summary.
// It uses the manager's LLM to analyze results across different workflows, identify
// patterns, resolve conflicts, and generate insights that might not be apparent
// when looking at individual Synergy results in isolation.
//
// The synthesis process:
//  1. Formats all Synergy results and the initial prompt
//  2. Constructs a meta-prompt for the LLM to analyze the results
//  3. Generates a comprehensive synthesis highlighting key insights
//
// Parameters:
//   - ctx: Context for cancellation and deadline control
//   - results: Combined results from all Synergies
//   - initialPrompt: The original user request for context
//
// Returns:
//   - map[string]interface{}: Synthesized insights and conclusions
//   - error: Any error encountered during synthesis
func (sm *SynergyManager) synthesizeResults(ctx context.Context, results map[string]interface{}, initialPrompt string) (map[string]interface{}, error) {
	resultString := fmt.Sprintf("Initial prompt: %s\n\nResults from Synergies:\n", initialPrompt)
	for k, v := range results {
		resultString += fmt.Sprintf("%s: %v\n", k, v)
	}

	prompt := fmt.Sprintf(
		"As a manager overseeing multiple AI Synergies, synthesize the following results into a cohesive summary. "+
			"Highlight key insights, resolve any conflicts, and provide an overall conclusion.\n\n%s",
		resultString,
	)

	response, err := gollm.LLM(sm.llm).Generate(ctx, gollm.NewPrompt(prompt))
	if err != nil {
		return nil, fmt.Errorf("error generating synthesis: %w", err)
	}

	return map[string]interface{}{"synthesis": response}, nil
}
