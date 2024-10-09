package enzu

import (
	"context"
	"fmt"

	"github.com/teilomillet/gollm"
)

// SynergyManager oversees and coordinates multiple Synergies
type SynergyManager struct {
	name      string
	llm       gollm.LLM
	synergies []*Synergy
	logger    *Logger
}

// NewSynergyManager creates a new SynergyManager
func NewSynergyManager(name string, llm gollm.LLM, logger *Logger) *SynergyManager {
	return &SynergyManager{
		name:      name,
		llm:       llm,
		synergies: make([]*Synergy, 0),
		logger:    logger,
	}
}

// AddSynergy adds a Synergy to the manager
func (sm *SynergyManager) AddSynergy(s *Synergy) {
	sm.synergies = append(sm.synergies, s)
}

// ExecuteSynergies runs all managed Synergies and coordinates their results
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

// synthesizeResults uses the manager's LLM to create a cohesive output from all Synergy results
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
