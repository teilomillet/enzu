// Package enzu implements a flexible and extensible tool system that enables
// agents to interact with external services and perform complex operations.
// The tool system provides:
//   - Dynamic tool registration and discovery
//   - Type-safe tool execution
//   - Tool organization through named lists
//   - Thread-safe concurrent access
//   - Validation of tool definitions
package enzu

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
)

// Tool represents an executable capability that can be used by agents within the framework.
// Tools are the primary mechanism for agents to interact with external systems, perform
// computations, or access services. Each tool has:
//   - A unique name for identification
//   - A description of its functionality
//   - An Execute function that implements the tool's behavior
//
// Tools can be organized into lists and shared across agents and synergies.
type Tool struct {
	// Name uniquely identifies the tool within the registry
	Name string

	// Description explains the tool's purpose and usage
	Description string

	// Execute implements the tool's functionality
	// Parameters:
	//   - args: Variable number of arguments specific to the tool
	// Returns:
	//   - interface{}: Tool-specific result
	//   - error: Any error encountered during execution
	Execute func(args ...interface{}) (interface{}, error)
}

// ToolList represents a named collection of related tools. Lists enable:
//   - Logical grouping of related tools
//   - Sharing common tools across agents
//   - Access control through tool availability
//   - Organization of domain-specific capabilities
type ToolList struct {
	// Name identifies the tool list
	Name string

	// Tools maps tool names to their implementations
	Tools map[string]Tool
}

// ToolRegistry manages the lifecycle and organization of tools within the framework.
// It provides:
//   - Centralized tool management
//   - Thread-safe tool access
//   - Tool list organization
//   - Tool discovery and retrieval
type ToolRegistry struct {
	// tools maps tool names to their implementations
	tools map[string]Tool

	// toolLists maps list names to ToolList instances
	toolLists map[string]*ToolList

	// mu ensures thread-safe access to the registry
	mu sync.Mutex
}

// NewToolRegistry creates a new ToolRegistry instance with an initialized
// default "Tools" list. The registry serves as the central repository for
// all tools in the framework.
//
// Returns:
//   - *ToolRegistry: A new registry instance ready for tool registration
func NewToolRegistry() *ToolRegistry {
	tr := &ToolRegistry{
		tools:     make(map[string]Tool),
		toolLists: make(map[string]*ToolList),
	}
	tr.toolLists["Tools"] = &ToolList{Name: "Tools", Tools: make(map[string]Tool)}
	return tr
}

// NewTool creates and registers a new tool with the default registry.
// It validates the tool definition and panics if validation fails.
//
// Parameters:
//   - name: Unique identifier for the tool
//   - description: Clear explanation of the tool's purpose
//   - execute: Function implementing the tool's behavior
//   - lists: Optional list names to add the tool to
//
// Returns:
//   - Tool: The created and registered tool instance
//
// Example:
//   calculator := NewTool(
//     "Calculator",
//     "Performs basic arithmetic operations",
//     func(args ...interface{}) (interface{}, error) {
//       // Implementation
//     },
//     "Math", "Basic"
//   )
func NewTool(name, description string, execute func(args ...interface{}) (interface{}, error), lists ...string) Tool {
	tool := Tool{
		Name:        name,
		Description: description,
		Execute:     execute,
	}

	if err := validateTool(tool); err != nil {
		panic(fmt.Sprintf("Invalid tool: %v", err))
	}

	defaultRegistry.registerTool(tool, lists...)

	return tool
}

// registerTool adds a tool to the registry and specified tool lists.
// If no lists are specified, the tool is added to the default "Tools" list.
//
// Parameters:
//   - tool: The tool to register
//   - lists: Optional list names to add the tool to
func (tr *ToolRegistry) registerTool(tool Tool, lists ...string) {
	tr.mu.Lock()
	defer tr.mu.Unlock()

	tr.tools[tool.Name] = tool

	if len(lists) == 0 {
		lists = []string{"Tools"}
	}

	for _, listName := range lists {
		if list, exists := tr.toolLists[listName]; exists {
			list.Tools[tool.Name] = tool
		} else {
			newList := &ToolList{
				Name:  listName,
				Tools: make(map[string]Tool),
			}
			newList.Tools[tool.Name] = tool
			tr.toolLists[listName] = newList
		}
	}
}

// GetTool retrieves a tool from the registry by its name.
//
// Parameters:
//   - name: The name of the tool to retrieve
//
// Returns:
//   - Tool: The requested tool
//   - bool: True if the tool exists, false otherwise
func (tr *ToolRegistry) GetTool(name string) (Tool, bool) {
	tool, exists := tr.tools[name]
	return tool, exists
}

// GetToolList retrieves a tool list from the registry by its name.
//
// Parameters:
//   - name: The name of the tool list to retrieve
//
// Returns:
//   - *ToolList: The requested tool list
//   - bool: True if the list exists, false otherwise
func (tr *ToolRegistry) GetToolList(name string) (*ToolList, bool) {
	list, exists := tr.toolLists[name]
	return list, exists
}

// ListTools returns a formatted list of all available tools
// and their descriptions. This is useful for:
//   - Tool discovery
//   - Documentation generation
//   - User interface presentation
//
// Returns:
//   - []string: List of tool descriptions in "name: description" format
func (tr *ToolRegistry) ListTools() []string {
	var toolList []string
	for _, tool := range tr.tools {
		toolList = append(toolList, fmt.Sprintf("%s: %s", tool.Name, tool.Description))
	}
	return toolList
}

// ListToolLists returns the names of all available tool lists
// in the registry. This enables:
//   - List discovery
//   - Tool organization overview
//   - Configuration validation
//
// Returns:
//   - []string: List of tool list names
func (tr *ToolRegistry) ListToolLists() []string {
	var listNames []string
	for name := range tr.toolLists {
		listNames = append(listNames, name)
	}
	return listNames
}

// validateTool ensures a tool definition meets the required schema:
//   - Name must be non-empty
//   - Description must be non-empty
//   - Execute must be a valid function
//   - Execute must return (interface{}, error)
//
// Parameters:
//   - tool: The tool to validate
//
// Returns:
//   - error: Validation error or nil if valid
func validateTool(tool Tool) error {
	if strings.TrimSpace(tool.Name) == "" {
		return fmt.Errorf("tool name cannot be empty")
	}
	if strings.TrimSpace(tool.Description) == "" {
		return fmt.Errorf("tool description cannot be empty")
	}
	if tool.Execute == nil {
		return fmt.Errorf("tool must have an Execute function")
	}
	executeType := reflect.TypeOf(tool.Execute)
	if executeType.Kind() != reflect.Func {
		return fmt.Errorf("Execute must be a function")
	}
	if executeType.NumOut() != 2 {
		return fmt.Errorf("Execute function must return two values (result and error)")
	}
	if !executeType.Out(1).Implements(reflect.TypeOf((*error)(nil)).Elem()) {
		return fmt.Errorf("second return value of Execute must be an error")
	}
	return nil
}

// defaultRegistry provides a package-level registry instance
// for convenient tool registration and access.
var defaultRegistry = NewToolRegistry()
