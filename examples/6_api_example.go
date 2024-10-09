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

type Server struct {
	synergy *enzu.Synergy
	logger  *enzu.Logger
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
		gollm.SetMaxTokens(200),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize LLM: %v", err)
	}

	// Create a logger for Enzu
	logger := enzu.NewLogger(enzu.LogLevelInfo)

	// Define options for the ExaSearch tool
	exaSearchOptions := tools.ExaSearchOptions{
		NumResults: 5,
		Type:       "neural",
		Contents: tools.Contents{
			Text: true,
		},
	UseAutoprompt:      true,
		StartPublishedDate: "2023-01-01T00:00:00.000Z",
	}

	// Register the ExaSearch tool
	tools.ExaSearch("", exaSearchOptions, "ResearchTool")

	// Create research agents
	researchAgent1 := enzu.NewAgent("Research Agent 1", "Agent specialized in AI research", llm,
		enzu.WithToolLists("ResearchTool"),
		enzu.WithParallelExecution(true),
	)
	researchAgent2 := enzu.NewAgent("Research Agent 2", "Agent specialized in startup research", llm,
		enzu.WithToolLists("ResearchTool"),
		enzu.WithParallelExecution(true),
	)

	// Create a synergy
	synergy := enzu.NewSynergy(
		"Parallel AI Research",
		llm,
		enzu.WithAgents(researchAgent1, researchAgent2),
		enzu.WithLogger(logger),
	)

	return &Server{
		synergy: synergy,
		logger:  logger,
	}, nil
}

func (s *Server) handleExecute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request struct {
		Tasks []string `json:"tasks"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Create tasks based on the request
	var tasks []*enzu.Task
	agents := s.synergy.GetAgents()
	for i, taskDescription := range request.Tasks {
		agent := agents[i%len(agents)] // Assign tasks to agents in a round-robin fashion
		tasks = append(tasks, enzu.NewTask(taskDescription, agent))
	}

	// Update synergy with new tasks
	s.synergy.SetTasks(tasks)

	// Execute the synergy
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	results, err := s.synergy.Execute(ctx)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error executing synergy: %v", err), http.StatusInternalServerError)
		return
	}

	// Send results back to the client
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func main() {
	server, err := NewServer()
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	http.HandleFunc("/execute", server.handleExecute)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Server starting on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
