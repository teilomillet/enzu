package enzu

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
)

// Tool represents a tool that can be used by agents
type Tool struct {
	Name        string
	Description string
	Execute     func(args ...interface{}) (interface{}, error)
}

// ToolList represents a named list of tools
type ToolList struct {
	Name  string
	Tools map[string]Tool
}

// ToolRegistry manages the available tools and tool lists
type ToolRegistry struct {
	tools     map[string]Tool
	toolLists map[string]*ToolList
	mu        sync.Mutex // Mutex to ensure thread safety
}

// NewToolRegistry creates a new ToolRegistry
func NewToolRegistry() *ToolRegistry {
	tr := &ToolRegistry{
		tools:     make(map[string]Tool),
		toolLists: make(map[string]*ToolList),
	}
	tr.toolLists["Tools"] = &ToolList{Name: "Tools", Tools: make(map[string]Tool)}
	return tr
}

// NewTool creates a new Tool and automatically registers it
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

// registerTool adds a new tool to the registry and specified ToolLists
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

// GetTool retrieves a tool from the registry by name
func (tr *ToolRegistry) GetTool(name string) (Tool, bool) {
	tool, exists := tr.tools[name]
	return tool, exists
}

// GetToolList retrieves a ToolList from the registry by name
func (tr *ToolRegistry) GetToolList(name string) (*ToolList, bool) {
	list, exists := tr.toolLists[name]
	return list, exists
}

// ListTools returns a list of all available tool names and descriptions
func (tr *ToolRegistry) ListTools() []string {
	var toolList []string
	for _, tool := range tr.tools {
		toolList = append(toolList, fmt.Sprintf("%s: %s", tool.Name, tool.Description))
	}
	return toolList
}

// ListToolLists returns a list of all available ToolList names
func (tr *ToolRegistry) ListToolLists() []string {
	var listNames []string
	for name := range tr.toolLists {
		listNames = append(listNames, name)
	}
	return listNames
}

// validateTool checks if the tool meets the required schema
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

// defaultRegistry is the package-level default ToolRegistry
var defaultRegistry = NewToolRegistry()
