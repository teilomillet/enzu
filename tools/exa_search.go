package tools

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/teilomillet/enzu"
)

// ExaSearchOptions represents the options for the ExaSearch tool
type ExaSearchOptions struct {
	Type               string
	UseAutoprompt      bool
	NumResults         int
	Contents           Contents
	StartPublishedDate string
}

// Contents represents the contents options for the Exa.ai API
type Contents struct {
	Text bool `json:"text"`
}

// ExaSearchResponse represents the response from the Exa.ai API
type ExaSearchResponse struct {
	RequestID          string            `json:"requestId"`
	AutopromptString   string            `json:"autopromptString"`
	ResolvedSearchType string            `json:"resolvedSearchType"`
	Results            []ExaSearchResult `json:"results"`
}

// ExaSearchResult represents a single result from the Exa.ai API
type ExaSearchResult struct {
	Score         float64 `json:"score"`
	Title         string  `json:"title"`
	ID            string  `json:"id"`
	URL           string  `json:"url"`
	PublishedDate string  `json:"publishedDate"`
	Author        string  `json:"author"`
	Text          string  `json:"text"`
}

// ExaSearchTool represents a tool for performing research using the Exa.ai API
type ExaSearchTool struct {
	APIKey  string
	Options ExaSearchOptions
	Logger  *enzu.Logger
	client  *http.Client
}

// NewExaSearchTool creates a new ExaSearchTool
func NewExaSearchTool(apiKey string, options ExaSearchOptions, logger *enzu.Logger) *ExaSearchTool {
	return &ExaSearchTool{
		APIKey:  apiKey,
		Options: options,
		Logger:  logger,
		client:  &http.Client{Timeout: 10 * time.Second},
	}
}

// Execute performs research queries using the Exa.ai API, supporting parallel execution
func (t *ExaSearchTool) Execute(args ...interface{}) (interface{}, error) {
	if len(args) < 1 {
		return nil, fmt.Errorf("ExaSearch tool requires at least one argument (query or queries)")
	}

	var queries []string
	var opts ExaSearchOptions

	// Handle different types of first argument
	switch arg := args[0].(type) {
	case string:
		queries = []string{arg}
	case []string:
		queries = arg
	case []interface{}:
		for _, q := range arg {
			if s, ok := q.(string); ok {
				queries = append(queries, s)
			} else {
				return nil, fmt.Errorf("all queries must be strings")
			}
		}
	default:
		return nil, fmt.Errorf("first argument must be a string or a slice of strings")
	}

	// Handle options if provided
	if len(args) > 1 {
		if o, ok := args[1].(ExaSearchOptions); ok {
			opts = o
		} else {
			opts = t.Options
		}
	} else {
		opts = t.Options
	}

	results := make([]interface{}, len(queries))
	errors := make([]error, len(queries))
	var wg sync.WaitGroup

	for i, query := range queries {
		wg.Add(1)
		go func(i int, query string) {
			defer wg.Done()
			result, err := t.executeQuery(query, opts)
			results[i] = result
			errors[i] = err
		}(i, query)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// executeQuery performs a single query to the Exa.ai API
func (t *ExaSearchTool) executeQuery(query string, opts ExaSearchOptions) (interface{}, error) {
	url := "https://api.exa.ai/search"

	body := struct {
		Query              string   `json:"query"`
		Type               string   `json:"type"`
		UseAutoprompt      bool     `json:"useAutoprompt"`
		NumResults         int      `json:"numResults"`
		Contents           Contents `json:"contents"`
		StartPublishedDate string   `json:"startPublishedDate,omitempty"`
	}{
		Query:              query,
		Type:               opts.Type,
		UseAutoprompt:      opts.UseAutoprompt,
		NumResults:         opts.NumResults,
		Contents:           opts.Contents,
		StartPublishedDate: opts.StartPublishedDate,
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		t.Logger.Error("ExaSearchTool", "failed to marshal request body: %v", err)
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		t.Logger.Error("ExaSearchTool", "failed to create HTTP request: %v", err)
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("accept", "application/json")
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-api-key", t.APIKey)

	resp, err := t.client.Do(req)
	if err != nil {
		t.Logger.Error("ExaSearchTool", "failed to send HTTP request: %v", err)
		return nil, fmt.Errorf("failed to send HTTP request: %w", err)
	}
	defer resp.Body.Close()

	var result ExaSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Logger.Error("ExaSearchTool", "failed to decode response: %v", err)
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	var results []struct {
		Title string `json:"title"`
		URL   string `json:"url"`
	}
	for _, res := range result.Results {
		results = append(results, struct {
			Title string `json:"title"`
			URL   string `json:"url"`
		}{
			Title: res.Title,
			URL:   res.URL,
		})
	}

	t.Logger.Info("ExaSearchTool", "Found %d results for query: %s", len(results), query)
	return results, nil
}

// ExaSearch returns a Tool for the ExaSearch tool with the given options
func ExaSearch(apiKey string, options ExaSearchOptions, lists ...string) enzu.Tool {
	if apiKey == "" {
		apiKey = os.Getenv("EXA_API_KEY")
		if apiKey == "" {
			panic("EXA_API_KEY environment variable is not set")
		}
	}

	logger := enzu.NewLogger(enzu.LogLevelDebug)
	exaTool := NewExaSearchTool(apiKey, options, logger)

	return enzu.NewTool(
		"ExaSearch",
		"Performs research using the Exa.ai API and returns the titles and URLs of the results. Supports parallel execution.",
		exaTool.Execute,
		lists...,
	)
}
