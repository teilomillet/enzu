# Tutorial: Building a Web Scraping Assistant

In this tutorial, we'll build a web scraping assistant that can fetch web pages, extract information, and save the results. This example demonstrates how to combine multiple agents with different responsibilities to create a useful application.

## What We'll Build

We'll create a system that can:
1. Fetch web pages
2. Extract specific information
3. Save the results to files
4. Handle errors gracefully

## Prerequisites

```go
go get github.com/teilomillet/enzu
go get github.com/PuerkitoBio/goquery  // For HTML parsing
```

## Step 1: Create the Tools

First, let's create the tools our agents will need:

```go
// Create network tools for fetching web pages
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

// Create parsing tools for extracting information
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
```

## Step 2: Create the Agent

Next, we'll create an agent that can use these tools:

```go
agent := enzu.NewAgent(
    "WebScrapingAgent",
    "An agent that can fetch and extract information from web pages",
    llm,
    enzu.WithToolLists("NetworkTools", "ParsingTools"),
)
```

## Step 3: Create Task and Synergy

Create a task for the agent and a synergy to manage execution:

```go
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
```

## Step 4: Execute the Synergy

Finally, execute the synergy to perform the web scraping:

```go
ctx := context.Background()
results, err := synergy.Execute(ctx)
if err != nil {
    panic(err)
}

fmt.Println("Extracted content:", results)
```

## Complete Example

For a complete working example, see [web_scraper_example.go](../../examples/web_scraper_example.go) in the examples directory. This example demonstrates all the concepts covered in this tutorial in a single, runnable file.

The example includes:
- Tool creation for URL fetching and text extraction
- Agent setup with appropriate tool lists
- Task and synergy creation
- Error handling and result output

Run the example with:
```bash
go run examples/web_scraper_example.go
```

## Running the Example

1. Save the code as `web_scraper.go`
2. Make sure your OpenAI API key is set:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
3. Run the program:
   ```bash
   go run web_scraper.go
   ```

## Understanding the Flow

1. The `WebScrapingAgent` agent uses the `FetchURL` tool to download the webpage
2. The `WebScrapingAgent` agent uses the `ExtractText` tool to find and extract paragraph text
3. The results are printed to the console

## Extending the Example

You can extend this example by:

1. Adding more parsing tools for different types of content
2. Creating a cleaning agent to process the extracted data
3. Adding error retry logic to the fetch tool
4. Implementing concurrent scraping of multiple pages

## Error Handling

The example includes basic error handling, but you might want to add:

1. Retry logic for failed requests
2. Input validation
3. Rate limiting
4. Response status code checking

## Best Practices

1. **Respect Websites**:
   - Check robots.txt
   - Implement rate limiting
   - Add proper User-Agent headers

2. **Error Handling**:
   - Handle network errors gracefully
   - Validate all inputs
   - Log errors for debugging

3. **Data Management**:
   - Clean and validate extracted data
   - Handle different character encodings
   - Store data in appropriate formats

4. **Performance**:
   - Use parallel execution when appropriate
   - Implement caching if needed
   - Monitor memory usage

Continue to the next tutorial to learn more advanced Enzu features!
