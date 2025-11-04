# OpenAI Streaming Tool Calls Implementation

A practical solution for handling streaming responses with OpenAI's API when using function/tool calls.

## 🌟 Key Features

- ✅ Stream both text responses AND tool/function calls from OpenAI's API
- ✅ Process multiple yield formats with a unified handling system
- ✅ Support for function composition (one function yielding from another)
- ✅ Maintain proper conversation history with complete context
- ✅ Handle multi-step conversations with proper chat continuation

## 🔍 The Problem This Solves

When working with OpenAI's streaming API and tool calls, several challenges emerge:

1. **Argument Streaming**: Arguments for tool calls come in as partial JSON chunks
2. **Yield Patterns**: There's no standard way to yield results from tool functions
3. **Chat History**: Tool calls and responses need proper integration in the conversation
4. **Continuation**: The model needs to see tool outputs to provide a summary/continuation

This implementation provides a complete solution for all these challenges.

## 💡 How It Works

### Core Components

1. **Stream Processing**:
   ```python
   stream = client.chat.completions.create(
       model="gpt-5-mini-2025-08-07",
       stream=True,
       messages=[...],
       tools=[fun_2_function],
       tool_choice="auto",
       temperature=1
   )
   ```
   - Listens for both text chunks and tool call chunks
   - Accumulates partial JSON arguments until complete

2. **Yield Patterns**:
   The implementation handles various yield formats:

   - **Plain Strings**: Simple text output
     ```python
     yield "LAST OF: fun_1"
     ```

   - **Display with Storage Flag**: Control what gets stored vs. displayed
     ```python
     yield {"display": "Text to show and store", "store": True}
     yield {"display": "Text to only show", "store": False}
     ```

   - **Nested Structures**: Complex data can be yielded
     ```python
     yield {"display": {"key": "value", "nested": {...}}, "store": True}
     ```
   
   - **Function Composition**: Yield from other functions
     ```python
     yield from fun_1()
     ```

   - **Edge Cases**: Handling None and empty dictionaries
     ```python
     yield None
     yield {}
     ```

3. **Yield Processing**:
   - Normalizes different output formats into a standard structure
   - Controls what's displayed to users vs. what's stored in history
   - Handles proper JSON serialization for complex objects

## 🔄 Conversation Flow

The implementation follows this process:

1. **User Message**: Added to conversation history
2. **Initial Streaming**: Get text or tool calls from the model
3. **Tool Execution**: When a complete tool call is received
   - Execute the appropriate function with the parsed arguments
   - Process and normalize each yield from the function
   - Display results to the user in real-time
4. **Tool Response**: Add the collected tool results to conversation history
5. **Continuation**: Get the model to continue with knowledge of tool results

## ⚙️ Yield Pattern Reference

| Yield Pattern | Display | Storage | Example |
|---------------|---------|---------|---------|
| Plain string | ✅ String is displayed | ❌ Not stored | `yield "Hello world"` |
| Dict with store=True | ✅ display field shown | ✅ Added to history | `yield {"display": "Save me", "store": True}` |
| Dict with store=False | ✅ display field shown | ❌ Not stored | `yield {"display": "Don't save", "store": False}` |
| Dict without store field | ✅ display field shown | ❌ Default: not stored | `yield {"display": "No store field"}` |
| Dict with nested display | ✅ JSON-formatted | ✅ If store=True | `yield {"display": {"key": "value"}, "store": True}` |
| Dict without display field | ✅ Other fields shown | ✅ If store=True | `yield {"message": "Error", "code": 404}` |
| None value | ❌ Nothing displayed | ❌ Not stored | `yield None` |
| Empty dict | ❌ Nothing displayed | ❌ Not stored | `yield {}` |

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenAI API key

### Installation

1. Clone the repository or download the files
2. Install the required dependency:
   ```bash
   pip install openai
   ```
3. Add your OpenAI API key to `config.py`

### Running the Demo

```bash
python openai_streaming_tools.py
```

Example interaction:
```
You: Calculate 5 + 10
Bot: I'll help you with that calculation using the fun_2 function.

fun_1fun_1fun_1fun_1

fun_2 using proper yield format ** should be stored in history and be showed in terminal **

fun_2 using proper yield format ** should not be stored in history and be showed in terminal **

{"DICT fun_2": "this should be stored in chat history", "success": true, "result": {"fun_2_string": "Hi this is fun_2", "fun_2_int": 67, "fun_2_float": 3.14, "fun_2_bool": true, "fun_2_dict": {"fun_2_test1": "test1", "fun_2_test2": "test2"}}}

result = 15

I've calculated 5 + 10 using the fun_2 function. The result is 15, as shown in the final output.
```

## 🔧 Implementation Details

### Conversation History Management

The conversation history is maintained with proper structure:
- User messages: `{"role": "user", "content": "..."}`
- Assistant tool calls: `{"role": "assistant", "tool_calls": [...]}`
- Tool responses: `{"role": "tool", "tool_call_id": "...", "content": "..."}`
- Assistant responses: `{"role": "assistant", "content": "..."}`

### JSON Argument Handling

Arguments for tool calls often come in pieces that need to be assembled:
```python
# Build up arguments as they stream in
partial_args += args_piece

try:
    # Try to parse the complete JSON arguments
    args = json.loads(partial_args)
    # Execute the tool function
    # ...
except json.JSONDecodeError:
    # Still receiving partial arguments
    pass
```

### Yield Normalization

The implementation normalizes different yield formats:
```python
if not isinstance(yield_part, dict):
    # Convert plain values to dicts with defaults
    yield_part = {
        "display": yield_part,
        "store": False
    }
else:
    # Ensure dict has all required fields with defaults
    if "store" not in yield_part:
        yield_part["store"] = False

    # Handle missing display field
    if "display" not in yield_part:
        # Create display from other fields
        display_data = {k: v for k, v in yield_part.items() if k not in ["store"]}
        if display_data:
            yield_part["display"] = display_data
        else:
            yield_part["display"] = ""
```

## 🛠 Customization

### Creating Your Own Tool Functions

1. Define a function that yields results:
   ```python
   def my_function(**args):
       # Process arguments
       yield "Starting work..."
       
       # Perform operations
       result = do_something(args)
       
       # Yield results in any supported format
       yield {"display": f"Result: {result}", "store": True}
   ```

2. Define the function signature for OpenAI:
   ```python
   my_function_definition = {
       "type": "function",
       "function": {
           "name": "my_function",
           "description": "Description of what your function does",
           "parameters": {
               "type": "object",
               "properties": {
                   # Your parameters here
               },
               "required": [/* required parameters */]
           },
       },
   }
   ```

3. Update the tools list when creating the chat completion:
   ```python
   tools=[my_function_definition],
   ```

## 📋 Best Practices

1. **Display vs. Store**:
   - Use `"store": True` only for information that should be part of the conversation context
   - Use `"store": False` for detailed outputs that don't need to be remembered

2. **Structured Outputs**:
   - For complex data, use nested dictionaries in the display field
   - This allows for both structured storage and formatted display

3. **Function Composition**:
   - Use `yield from other_function()` to compose functions
   - This maintains consistent handling of all yield patterns

4. **Error Handling**:
   - Always wrap tool execution in try/except blocks
   - Yield error messages with a descriptive format

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
