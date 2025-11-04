from openai import OpenAI
import json
import time

import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def fun_1():
    """
    Example function that yields different types of outputs.

    Yields:
        Various formats to demonstrate streaming response handling
    """
    yield {"display": f"{'fun_1' * 4}", "store": True}

    yield f"LAST OF: fun_1"


def fun_2(a: int, b: int):
    """
    Another example function that shows function composition and
    various yield formats.

    Args:
        a (int): First number
        b (int): Second number

    Yields:
        Various formats to demonstrate streaming response handling
    """
    # Call another function and yield its results
    yield from fun_1()

    # Print to terminal only (not visible to the user)
    print(f"fun_2 with a={a}, b={b} ** only print in terminal **")

    # Plain string yield
    yield "fun_2 plain string test ** only be showed in terminal **"

    # Dictionary with display and store=True
    yield {"display": "fun_2 using proper yield format ** should be stored in history and be showed in terminal **",
           "store": True}

    # Dictionary with display and store=False
    yield {"display": "fun_2 using proper yield format ** should not be stored in history and be showed in terminal **",
           "store": False}

    # Complex nested dictionary
    dict = {
        "fun_2_string": "Hi this is fun_2",
        "fun_2_int": 67,
        "fun_2_float": 3.14,
        "fun_2_bool": True,
        "fun_2_dict": {
            "fun_2_test1": "test1",
            "fun_2_test2": "test2"
        }
    }

    # Dictionary with complex nested structure
    yield {"display": {"DICT fun_2": "this should be stored in chat history", "success": True, "result": dict},
           "code": 200, "store": True}

    # Edge cases
    yield None
    yield {}

    yield {"display": f"result = {a+b}", "store": True}


fun_2_function = {
    "type": "function",
    "function": {
        "name": "fun_2",
        "description": "Add two integers and stream partial chunks (simulation).",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "First integer to add"
                },
                "b": {
                    "type": "integer",
                    "description": "Second integer to add"
                }
            },
            "required": ["a", "b"]
        },
    },
}


history = []


def ask_question(ask: str):
    """
    Process user input, make API calls to OpenAI, and handle tool execution

    Args:
        user_input: User's question or command

    Yields:
        Chunks of the response as they are generated
    """
    history.append({"role": "user", "content": ask})
    # To visual the yields added to history
    # print(history)

    stream = client.chat.completions.create(
        model="gpt-5-mini-2025-08-07",
        stream=True,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant. Add a short explanatory line after tool calls."},
            *history,
        ],
        tools=[fun_2_function],
        tool_choice="auto",
        temperature=1
    )

    collected = ""
    partial_args = ""
    tool_output = None
    tool_name = None
    tool_output_chunks = []
    try:
        # Phase 1: Listen for text or tool calls
        for chunk in stream:
            if not (hasattr(chunk, "choices") and chunk.choices):
                continue

            choice = chunk.choices[0]
            delta = getattr(choice.delta, "content", None)

            # Handle normal model text
            if delta:
                yield delta
                collected += delta

            # Handle tool/function call stream
            if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    # Extract tool call information
                    tool_name = getattr(tool_call.function, "name", None) or "worlds"
                    tool_id = getattr(tool_call, "id", None) or f"call_{int(time.time() * 1000)}"
                    args_piece = getattr(tool_call.function, "arguments", "")
                    if not args_piece:
                        continue

                    # Build up arguments as they stream in
                    partial_args += args_piece

                    try:
                        # Try to parse the complete JSON arguments
                        args = json.loads(partial_args)

                        # Log the assistant's tool call message
                        history.append({
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(args)
                                    },
                                }
                            ],
                        })

                        # Process each yield from the tool
                        for yield_part in fun_2(**args):
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
                                    # Create display from other fields (excluding type and store)
                                    display_data = {k: v for k, v in yield_part.items() if k not in ["store"]}

                                    if display_data:
                                        yield_part["display"] = display_data
                                    else:
                                        yield_part["display"] = ""

                            display_value = yield_part["display"]

                            if display_value:
                                yield display_value

                            # Store in output if requested
                            if yield_part.get("store", False):
                                # Store with type information for better processing later
                                tool_output_chunks.append(display_value)

                        tool_output_str = json.dumps(tool_output_chunks, ensure_ascii=False)

                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_id,  # required link
                            "name": tool_name,
                            "content": tool_output_str,
                        })
                        partial_args = ""  # reset
                    except json.JSONDecodeError:
                        # still receiving partial args
                        pass

        # Phase 2: Send tool output back for summary/continuation if needed
        if tool_output_chunks and tool_name:

            # Create a follow-up streaming completion with tool results
            followup = client.chat.completions.create(
                model="gpt-4o-mini",
                stream=True,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant. Add a short explanatory line after tool calls."},
                    *history,
                ],
                temperature=1,
            )

            # Stream the follow-up response
            for chunk in followup:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
                        collected += delta

    except KeyboardInterrupt:
        yield "\n[INTERRUPTED]\n"
    finally:
        history.append({"role": "assistant", "content": collected})


if __name__ == "__main__":
    """Main interactive loop to demonstrate the functionality"""
    print("OpenAI Streaming Tool Calls Demo")
    print("Type 'exit' or 'quit' to end the session\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        print("Bot:", end=" ", flush=True)
        for chunk in ask_question(user_input):
            print(chunk, end="", flush=True)
        print("\n")
