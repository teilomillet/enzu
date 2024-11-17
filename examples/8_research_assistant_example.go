package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/teilomillet/enzu"
	"github.com/teilomillet/enzu/tools"
	"github.com/teilomillet/gollm"
)

// Server represents our HTTP server with AI capabilities
type Server struct {
	manager *enzu.SynergyManager
	logger  *enzu.Logger
	llm     gollm.LLM
}

// ResearchRequest represents the structure of research requests
type ResearchRequest struct {
	Topic       string   `json:"topic"`
	Subtopics   []string `json:"subtopics,omitempty"`
	MaxResults  int      `json:"max_results,omitempty"`
	TimeoutSecs int      `json:"timeout_secs,omitempty"`
}

// ResearchResponse represents the structure of research responses
type ResearchResponse struct {
	MainTopicResults map[string]interface{} `json:"main_topic_results"`
	SubtopicResults  map[string]interface{} `json:"subtopic_results,omitempty"`
	ExecutionTime    float64                `json:"execution_time_seconds"`
}

func NewServer() (*Server, error) {
	// Load API key from environment variable
	openAIKey := os.Getenv("OPENAI_API_KEY")
	if openAIKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}

	// Create a new LLM instance
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(openAIKey),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(time.Second*2),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize LLM: %v", err)
	}

	// Create a logger for Enzu
	logger := enzu.NewLogger(enzu.LogLevelInfo)

	// Register the ExaSearch tool
	exaSearchOptions := tools.ExaSearchOptions{
		NumResults: 5,
		Type:       "neural",
		Contents: tools.Contents{
			Text: true,
		},
		UseAutoprompt:      true,
		StartPublishedDate: "2023-01-01T00:00:00.000Z",
	}
	tools.ExaSearch("", exaSearchOptions, "ResearchTool")

	// Create SynergyManager
	manager := enzu.NewSynergyManager("Advanced Research Assistant", llm, logger)

	return &Server{
		manager: manager,
		logger:  logger,
		llm:     llm,
	}, nil
}

func createResearchSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
	agent1 := enzu.NewAgent("Primary Research", "Specialized in deep research and fact verification", llm,
		enzu.WithToolLists("ResearchTool"),
		enzu.WithParallelExecution(true),
	)
	agent2 := enzu.NewAgent("Analysis Research", "Specialized in data analysis and insights", llm,
		enzu.WithToolLists("ResearchTool"),
		enzu.WithParallelExecution(true),
	)

	return enzu.NewSynergy(
		"Research Team",
		llm,
		enzu.WithAgents(agent1, agent2),
		enzu.WithLogger(logger),
	)
}

func createAnalysisSynergy(llm gollm.LLM, logger *enzu.Logger) *enzu.Synergy {
	agent := enzu.NewAgent("Data Analyst", "Specialized in processing and analyzing research results", llm,
		enzu.WithToolLists("ResearchTool"),
		enzu.WithParallelExecution(true),
	)

	return enzu.NewSynergy(
		"Analysis Team",
		llm,
		enzu.WithAgents(agent),
		enzu.WithLogger(logger),
	)
}

func (s *Server) handleResearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.logger.Error("API", "Invalid method: %s, expected POST", r.Method)
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ResearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.logger.Error("API", "Failed to decode request body: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	s.logger.Info("API", "Received research request for topic: %s with %d subtopics", 
		req.Topic, len(req.Subtopics))

	// Set defaults if not provided
	if req.MaxResults == 0 {
		req.MaxResults = 5
		s.logger.Debug("API", "Using default MaxResults: %d", req.MaxResults)
	}
	if req.TimeoutSecs == 0 {
		req.TimeoutSecs = 30
		s.logger.Debug("API", "Using default TimeoutSecs: %d", req.TimeoutSecs)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(r.Context(), time.Duration(req.TimeoutSecs)*time.Second)
	defer cancel()

	// Create tasks for research synergy
	s.logger.Info("Research", "Creating research synergy for main topic: %s", req.Topic)
	researchSynergy := createResearchSynergy(s.llm, s.logger)
	mainTask := enzu.NewTask(fmt.Sprintf("Research thoroughly: %s", req.Topic), researchSynergy.GetAgents()[0])
	researchSynergy.AddTask(mainTask)
	
	// Add subtopic tasks if any
	if len(req.Subtopics) > 0 {
		s.logger.Info("Research", "Adding %d subtopic tasks", len(req.Subtopics))
		for _, subtopic := range req.Subtopics {
			s.logger.Debug("Research", "Adding subtopic task: %s", subtopic)
			subtask := enzu.NewTask(
				fmt.Sprintf("Research subtopic: %s in context of %s", subtopic, req.Topic), 
				researchSynergy.GetAgents()[1],
			)
			researchSynergy.AddTask(subtask)
		}
	}

	// Create analysis synergy with task
	s.logger.Info("Analysis", "Creating analysis synergy")
	analysisSynergy := createAnalysisSynergy(s.llm, s.logger)
	analysisTask := enzu.NewTask(fmt.Sprintf("Analyze and synthesize research results for topic '%s' and subtopics %v", req.Topic, req.Subtopics), analysisSynergy.GetAgents()[0])
	analysisSynergy.AddTask(analysisTask)

	// Add both synergies to manager
	s.logger.Info("Manager", "Adding synergies to manager")
	s.manager.AddSynergy(researchSynergy)
	s.manager.AddSynergy(analysisSynergy)

	// Execute all synergies
	startTime := time.Now()
	s.logger.Info("Manager", "Starting execution of all synergies")
	results, err := s.manager.ExecuteSynergies(ctx, req.Topic)
	if err != nil {
		s.logger.Error("Manager", "Failed to execute synergies: %v", err)
		http.Error(w, fmt.Sprintf("Error executing synergies: %v", err), http.StatusInternalServerError)
		return
	}

	executionTime := time.Since(startTime).Seconds()
	s.logger.Info("Manager", "Synergies execution completed in %.2f seconds", executionTime)

	// Log results details
	if results != nil {
		s.logger.Info("Results", "Number of result keys: %d", len(results))
		for key := range results {
			s.logger.Debug("Results", "Found result key: %s", key)
		}
		
		if synthesis, ok := results["synthesis"]; ok {
			// Log first 100 characters of synthesis for preview
			synthStr, _ := synthesis.(string)
			previewLen := 100
			if len(synthStr) < previewLen {
				previewLen = len(synthStr)
			}
			s.logger.Info("Results", "Synthesis preview: %s...", synthStr[:previewLen])
		}
	}

	// Prepare response
	response := ResearchResponse{
		MainTopicResults: results,
		SubtopicResults:  nil, // Results are now combined in MainTopicResults
		ExecutionTime:    executionTime,
	}

	// Log response details
	responseJSON, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		s.logger.Error("API", "Failed to marshal response: %v", err)
	} else {
		s.logger.Debug("API", "Full response:\n%s", string(responseJSON))
	}

	s.logger.Info("API", "Sending response with %d main results", len(results))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	server, err := NewServer()
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	// Set logger to debug level for more detailed logs
	server.logger.SetLevel(enzu.LogLevelDebug)

	// Register routes
	http.HandleFunc("/research", server.handleResearch)

	// Start the server
	port := ":8080"
	fmt.Printf("Server starting on port %s...\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
