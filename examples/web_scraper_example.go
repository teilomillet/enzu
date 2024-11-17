package main

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"github.com/teilomillet/enzu"
	"github.com/teilomillet/gollm"
)

func main() {
	// Initialize LLM
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
	)
	if err != nil {
		panic(err)
	}

	// Create FetchURL tool
	enzu.NewTool(
		"FetchURL",
		"Fetches content from a URL",
		func(args ...interface{}) (interface{}, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("FetchURL requires 1 argument (url)")
			}
			url, ok := args[0].(string)
			if !ok {
				return nil, fmt.Errorf("URL must be a string")
			}

			resp, err := http.Get(url)
			if err != nil {
				return nil, err
			}
			defer resp.Body.Close()

			doc, err := goquery.NewDocumentFromReader(resp.Body)
			if err != nil {
				return nil, err
			}

			return doc.Html()
		},
		"NetworkTools",
	)

	// Create ExtractText tool
	enzu.NewTool(
		"ExtractText",
		"Extracts text content from HTML using CSS selector",
		func(args ...interface{}) (interface{}, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("ExtractText requires 2 arguments (html, selector)")
			}
			html, ok1 := args[0].(string)
			selector, ok2 := args[1].(string)
			if !ok1 || !ok2 {
				return nil, fmt.Errorf("Arguments must be strings")
			}

			doc, err := goquery.NewDocumentFromReader(strings.NewReader(html))
			if err != nil {
				return nil, err
			}

			var results []string
			doc.Find(selector).Each(func(i int, s *goquery.Selection) {
				results = append(results, strings.TrimSpace(s.Text()))
			})

			return results, nil
		},
		"ParsingTools",
	)

	// Create web scraping agent
	agent := enzu.NewAgent(
		"WebScrapingAgent",
		"An agent that can fetch and extract information from web pages",
		llm,
		enzu.WithToolLists("NetworkTools", "ParsingTools"),
	)

	// Create task
	task := enzu.NewTask(
		"Fetch content from example.com and extract all paragraph text",
		agent,
	)

	// Create synergy
	synergy := enzu.NewSynergy(
		"Web Scraping Example",
		llm,
		enzu.WithAgents(agent),
		enzu.WithTasks(task),
	)

	// Execute the synergy
	ctx := context.Background()
	results, err := synergy.Execute(ctx)
	if err != nil {
		panic(err)
	}

	fmt.Println("Extracted content:", results)
}
