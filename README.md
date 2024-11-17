# Enzu: Multi-Agent Framework for AI Systems

[![Go Reference](https://pkg.go.dev/badge/github.com/teilomillet/enzu.svg)](https://pkg.go.dev/github.com/teilomillet/enzu)
[![Go Report Card](https://goreportcard.com/badge/github.com/teilomillet/enzu)](https://goreportcard.com/report/github.com/teilomillet/enzu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Enzu is a declarative Go framework designed for building sophisticated multi-agent AI systems. It enables LLMs and AI agents to collaborate, execute parallel tasks, and leverage extensible tools while maintaining clear hierarchies and communication patterns.

## 🎯 Framework Capabilities

### Agent System Architecture
- **Hierarchical Agent Organization**: Define agent roles, responsibilities, and relationships
- **Dynamic Task Distribution**: Automatically route tasks to specialized agents
- **Parallel Processing**: Execute multiple agent tasks concurrently
- **State Management**: Track and maintain agent states across interactions

### Tool Integration System
- **Declarative Tool Registry**: Register and manage tools with clear interfaces
- **Inheritance Patterns**: Tools can be inherited and shared across agent hierarchies
- **Thread-Safe Operations**: Concurrent tool access with built-in safety mechanisms
- **Custom Tool Creation**: Extend functionality through a standardized tool interface

### Execution Patterns
- **Synergy-Based Collaboration**: Group agents into task-focused collaborative units
- **Context Propagation**: Share context and state across agent boundaries
- **Parallel Task Execution**: Optimize performance through concurrent processing
- **Error Recovery**: Built-in retry mechanisms and error handling patterns

### Communication Infrastructure
- **HTTP Server Integration**: Built-in REST API capabilities
- **Structured Message Passing**: Type-safe communication between agents
- **Event System**: Publish-subscribe patterns for agent coordination
- **Logging System**: Comprehensive tracing and debugging capabilities

## 🔧 Core Integration Patterns

### 1. Research and Analysis Pattern
```go
// Pattern: Distributed Research System
type ResearchRequest struct {
    Topic       string   `json:"topic"`
    Subtopics   []string `json:"subtopics,omitempty"`
    MaxResults  int      `json:"max_results,omitempty"`
    TimeoutSecs int      `json:"timeout_secs,omitempty"`
}

// Create specialized research agents
researcher := enzu.NewAgent("Primary Researcher",
    "Deep research and fact verification",
    llm,
    enzu.WithToolLists("ResearchTool"),
    enzu.WithParallelExecution(true),
)

analyst := enzu.NewAgent("Data Analyst",
    "Process and analyze research results",
    llm,
    enzu.WithToolLists("AnalysisTool"),
    enzu.WithParallelExecution(true),
)
```

### 2. Self-Aware System Pattern
```go
// Pattern: Self-Aware Interactive System
manager := enzu.NewSynergyManager("Self-Aware System", llm, logger)

// Define capability domains
researchSynergy := createDomainSynergy("Research", llm, logger)
analysisSynergy := createDomainSynergy("Analysis", llm, logger)
creativeSynergy := createDomainSynergy("Creative", llm, logger)

// Register domains
manager.AddSynergy(researchSynergy)
manager.AddSynergy(analysisSynergy)
manager.AddSynergy(creativeSynergy)
```

### 3. Tool Integration Pattern
```go
// Pattern: Extensible Tool System
exaSearchOptions := tools.ExaSearchOptions{
    NumResults: 5,
    Type:      "neural",
    Contents: tools.Contents{
        Text: true,
    },
    UseAutoprompt:      true,
    StartPublishedDate: "2023-01-01T00:00:00.000Z",
}
tools.RegisterTool("ResearchTool", exaSearchOptions)
```

### 4. API Integration Pattern
```go
// Pattern: Multi-Agent API Server
type Server struct {
    synergy *enzu.Synergy
    logger  *enzu.Logger
}

// Initialize server with parallel processing capabilities
func NewServer() (*Server, error) {
    // Create research agents with specific roles
    researchAgent1 := enzu.NewAgent("Research Agent 1",
        "Agent specialized in AI research",
        llm,
        enzu.WithToolLists("ResearchTool"),
        enzu.WithParallelExecution(true),
    )
    researchAgent2 := enzu.NewAgent("Research Agent 2",
        "Agent specialized in startup research",
        llm,
        enzu.WithToolLists("ResearchTool"),
        enzu.WithParallelExecution(true),
    )

    // Create parallel processing synergy
    synergy := enzu.NewSynergy(
        "Parallel AI Research",
        llm,
        enzu.WithAgents(researchAgent1, researchAgent2),
        enzu.WithLogger(logger),
    )

    return &Server{synergy: synergy, logger: logger}, nil
}

// Handle parallel task execution
func (s *Server) handleExecute(w http.ResponseWriter, r *http.Request) {
    var request struct {
        Tasks []string `json:"tasks"`
    }
    
    // Distribute tasks among agents
    agents := s.synergy.GetAgents()
    for i, taskDescription := range request.Tasks {
        agent := agents[i%len(agents)] // Round-robin distribution
        tasks = append(tasks, enzu.NewTask(taskDescription, agent))
    }
}

## 🚀 Capability Domains

### 1. Research & Information Retrieval
- Neural search integration (`ExaSearch` tool)
- Multi-agent research coordination
- Parallel information gathering
- Research result synthesis

### 2. Task Management & Execution
- Multi-agent task distribution
- Parallel task execution
- Progress tracking
- Result aggregation

### 3. Web Content Processing
- URL content fetching (`FetchURL` tool)
- HTML parsing and extraction
- CSS selector-based targeting
- Structured data collection

### 4. Synergy Management
- Multi-synergy orchestration
- Result synthesis across synergies
- Team-based agent organization
- Cross-team coordination
- Hierarchical task execution

### 5. Team Organization
- Role-specialized agents
- Team-based synergies
- Domain-specific agent groups
- Task-team alignment

### 6. API Integration & Scaling
- **Parallel Task Distribution**
  - Round-robin task assignment
  - Load-balanced processing
  - Concurrent execution
  - Real-time response handling

- **HTTP Service Integration**
  - RESTful endpoints
  - JSON request/response
  - Error handling patterns
  - Status monitoring

- **Multi-Agent Coordination**
  - Role-based agent assignment
  - Task synchronization
  - Result aggregation
  - State management

## 📦 Installation

```bash
go get github.com/teilomillet/enzu
```

## 📚 Integration Resources

### Core Documentation
- `/docs`: Architecture and integration guides
- `/docs/tutorials`: Step-by-step implementation patterns
- `/examples`: Reference implementations and use cases

### Example Implementations
1. Research Assistant System (`examples/8_research_assistant_example.go`)
2. Self-Aware System (`examples/7_manager_mode_example.go`)
3. Parallel Processing System (`examples/4_parallel_example.go`)
4. Tool Integration System (`examples/3_tools_example.go`)
5. API Integration System (`examples/6_api_example.go`)

### Integration Patterns
1. **HTTP API Integration**
   - REST endpoint creation
   - Request/Response handling
   - Timeout management
   - Error recovery
   - Round-robin task distribution
   - Load balancing strategies

2. **Tool Registry Integration**
   - Tool registration
   - Capability inheritance
   - Access control
   - Resource management

3. **Agent Collaboration**
   - Task distribution
   - Result synthesis
   - Context sharing
   - Error handling
