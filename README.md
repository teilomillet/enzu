# enzu

# Enzu

[![Go Reference](https://pkg.go.dev/badge/github.com/teilomillet/enzu.svg)](https://pkg.go.dev/github.com/teilomillet/enzu)
[![Go Report Card](https://goreportcard.com/badge/github.com/teilomillet/enzu)](https://goreportcard.com/report/github.com/teilomillet/enzu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Enzu is a Go framework for building AI applications with multi-agent collaboration, parallel task execution, and extensible tool integration. It enables the creation of complex AI systems that can work together to solve problems more effectively than single agents.

## 🌟 Key Features

- **Multi-Agent Synergies**: Create and manage groups of specialized AI agents that work together towards common objectives
- **Parallel Task Execution**: Execute tasks concurrently for improved performance
- **Flexible Tool Integration**: Extensible tool system with built-in registry and inheritance
- **Comprehensive Logging**: Detailed logging system with multiple levels of verbosity
- **HTTP Server Support**: Built-in HTTP server capabilities for creating API endpoints
- **LLM Integration**: Seamless integration with Large Language Models (currently supporting OpenAI)

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

```bash
go get github.com/teilomillet/enzu
```

## 🎯 Quick Start

Here's a simple example to get you started:

```go
package main

import (
    "context"
    "log"
    "os"
    "github.com/teilomillet/enzu"
    "github.com/teilomillet/gollm"
)

func main() {
    // Initialize LLM
    llm, err := gollm.NewLLM(
        gollm.SetProvider("openai"),
        gollm.SetModel("gpt-4o-mini"),
        gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Create agents
    analyst := enzu.NewAgent("Analyst", "Analyze data", llm)
    strategist := enzu.NewAgent("Strategist", "Develop strategies", llm)

    // Create synergy
    synergy := enzu.NewSynergy(
        "Market Analysis",
        llm,
        enzu.WithAgents(analyst, strategist),
        enzu.WithTasks(
            enzu.NewTask("Analyze market trends", analyst),
            enzu.NewTask("Develop strategy", strategist),
        ),
    )

    // Execute
    results, err := synergy.Execute(context.Background())
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Results: %v", results)
}
```

## 🧩 Core Concepts

### Agents
Agents are specialized AI entities with specific roles and capabilities. Each agent can:
- Access specific tools
- Execute tasks in parallel
- Inherit tools from their parent synergy

### Synergies
Synergies are collaborative groups of agents working together. They provide:
- Task coordination
- Resource sharing
- Result synthesis

### Tools
Tools are reusable functions that agents can leverage:
- Organized in tool lists
- Inheritance system
- Thread-safe registry

### Tasks
Tasks represent specific work items that need to be completed:
- Assigned to specific agents
- Can be executed in parallel
- Support context sharing

## 📚 Examples

### Creating a Tool
```go
enzu.NewTool(
    "CurrentTime",
    "Returns the current time",
    func(args ...interface{}) (interface{}, error) {
        return time.Now().Format(time.RFC3339), nil
    },
    "TimeTool",
)
```

### Parallel Execution
```go
agent := enzu.NewAgent("FastAgent", "Executes tasks in parallel", llm,
    enzu.WithParallelExecution(true),
)
```

### HTTP Server
```go
server := NewServer()
http.HandleFunc("/execute", server.handleExecute)
http.ListenAndServe(":8080", nil)
```

