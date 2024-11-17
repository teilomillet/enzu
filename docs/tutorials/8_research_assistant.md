# Enzu Research Assistant Tutorial

This tutorial demonstrates how to build a comprehensive research assistant using the Enzu library. The example combines multiple Enzu features including synergy management, parallel task execution, and AI-powered research capabilities.

## Overview

The research assistant is implemented as an HTTP server that:
1. Accepts research requests with a main topic and optional subtopics
2. Performs parallel research on all topics using AI agents
3. Synthesizes the results into a coherent summary
4. Returns the findings via a JSON response

## Key Components

### 1. Server Structure
```go
type Server struct {
    manager *enzu.SynergyManager
    logger  *enzu.Logger
    llm     gollm.LLM
}
```
- `manager`: Coordinates multiple synergies and their execution
- `logger`: Provides structured logging
- `llm`: Handles LLM interactions

### 2. Request/Response Format
```go
type ResearchRequest struct {
    Topic       string   `json:"topic"`
    Subtopics   []string `json:"subtopics,omitempty"`
    MaxResults  int      `json:"max_results,omitempty"`
    TimeoutSecs int      `json:"timeout_secs,omitempty"`
}

type ResearchResponse struct {
    MainTopicResults map[string]interface{} `json:"main_topic_results"`
    SubtopicResults  map[string]interface{} `json:"subtopic_results,omitempty"`
    ExecutionTime    float64                `json:"execution_time_seconds"`
}
```

## Implementation Steps

### 1. Create Research Synergy
The research synergy is responsible for gathering information:
```go
func createResearchSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
    synergy := enzu.NewSynergy("Research Team")
    
    // Add research agents with different roles
    primaryResearcher := enzu.NewAgent("Primary Research", "Specialized in deep research and fact verification", llm)
    analysisResearcher := enzu.NewAgent("Analysis Research", "Specialized in data analysis and insights", llm)
    
    synergy.AddAgent(primaryResearcher)
    synergy.AddAgent(analysisResearcher)
    
    return synergy
}
```

### 2. Create Analysis Synergy
The analysis synergy synthesizes the research results:
```go
func createAnalysisSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
    synergy := enzu.NewSynergy("Analysis Team")
    
    analyst := enzu.NewAgent("Data Analyst", "Specialized in processing and analyzing research results", llm)
    synergy.AddAgent(analyst)
    
    return synergy
}
```

### 3. Handle Research Requests
The main research process:
1. Create research synergy for main topic
2. Add subtopic tasks if provided
3. Create analysis synergy for result synthesis
4. Execute all synergies in parallel
5. Process and return results

## Example Usage

1. Start the server:
```bash
go run examples/8_research_assistant_example.go
```

2. Send a research request:
```bash
curl -X POST http://localhost:8080/research \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Latest advancements in quantum computing",
    "subtopics": ["quantum supremacy", "error correction"],
    "max_results": 5,
    "timeout_secs": 30
  }'
```

3. Example Response:
```json
{
  "main_topic_results": {
    "synthesis": "### Summary of Latest Advancements in Quantum Computing\n\n#### Key Insights\n\n1. **Advancements in Quantum Processing**:\n   - Reconfigurable Atom Arrays\n   - Universal Logic with Encoded Spin Qubits\n   ..."
  },
  "execution_time_seconds": 24.02
}
```

## Key Features

1. **Parallel Execution**: Multiple research tasks run concurrently
2. **Structured Logging**: Detailed logging at INFO and DEBUG levels
3. **Flexible Configuration**: Configurable timeouts and result limits
4. **Error Handling**: Robust error handling throughout the process
5. **Result Synthesis**: AI-powered analysis of research findings

## Best Practices

1. **Error Handling**: Always check for and handle errors appropriately
2. **Logging**: Use structured logging for better debugging
3. **Timeouts**: Set reasonable timeouts for research tasks
4. **Rate Limiting**: Consider implementing rate limiting for production use
5. **Response Size**: Monitor and potentially limit response sizes

## Advanced Usage

### Custom Agent Roles
You can customize agent roles for specific research needs:
```go
customAgent := enzu.NewAgent(
    "Domain Expert",
    "Specialized in specific domain knowledge",
    llm,
)
```

### Enhanced Error Handling
Add custom error handling for specific scenarios:
```go
if err := synergy.Execute(ctx); err != nil {
    switch err.(type) {
    case *enzu.TimeoutError:
        // Handle timeout
    case *enzu.APIError:
        // Handle API errors
    default:
        // Handle other errors
    }
}
```

## Conclusion

This example demonstrates the power of combining multiple Enzu features to create a sophisticated research assistant. The modular design allows for easy customization and extension while maintaining robust error handling and logging.
