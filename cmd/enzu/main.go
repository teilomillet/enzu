// Package main provides the entry point for the Enzu framework, a powerful tool for building
// and executing AI-driven workflows. It handles configuration loading and orchestrates the
// creation and execution of synergies based on provided YAML or JSON configurations.
package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/teilomillet/enzu/builder"
)

// main is the entry point of the Enzu framework. It performs the following steps:
//  1. Parses command-line flags to get the configuration file path
//  2. Loads and validates the synergy configuration
//  3. Creates a builder instance to construct the synergy
//  4. Builds and executes the synergy with the specified configuration
//
// The program expects a configuration file path provided via the -config flag.
// It will exit with an error if:
//  - No configuration file is provided
//  - The configuration file cannot be loaded or is invalid
//  - Synergy building fails
//  - Synergy execution encounters an error
func main() {
	configFile := flag.String("config", "", "Path to the configuration file (YAML or JSON)")
	flag.Parse()

	if *configFile == "" {
		log.Fatal("Please provide a configuration file using -config flag")
	}

	// Load configuration
	config, err := builder.LoadConfig(*configFile)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Create builder
	b := builder.NewBuilder(config)

	// Build synergy
	synergy, err := b.BuildSynergy()
	if err != nil {
		log.Fatalf("Failed to build synergy: %v", err)
	}

	// Execute synergy
	result, err := synergy.Execute(nil)
	if err != nil {
		log.Fatalf("Failed to execute synergy: %v", err)
	}

	fmt.Printf("Synergy execution completed. Result: %v\n", result)
}
