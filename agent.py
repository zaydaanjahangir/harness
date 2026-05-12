import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# Use whatever model model works, streaming only works on paid models
MODEL = "deepseek/deepseek-v4-flash"

YOU_COLOR = "\033[94m"  # bright blue
TOOL_COLOR = "\033[92m"  # bright green
ASSISTANT_COLOR = "\033[93m"  # bright yellow
RESET = "\033[0m"  # reset


SYSTEM_PROMPT = """You are a coding assistant that helps solve coding tasks.
You have access to tools you can execute:

{tool_list_repr}

When you want to use a tool, reply with exactly one line in the format:
  tool: TOOL_NAME({{"key": "value"}})
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.
"""


# --- Tools ---
def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def read_file_tool(filename: str) -> Dict[str, Any]:
    """
    Gets the full content of a file.
    :param filename: The path of the file to read.
    :return: The full content of the file.
    """
    path = resolve_path(filename)
    return {"file_path": str(path), "content": path.read_text(encoding="utf-8")}


def list_files_tool(path: str = ".") -> Dict[str, Any]:
    """
    Lists files and directories at a given path. Defaults to current directory.
    :param path: The directory path to list.
    :return: A list of files and directories.
    """
    full_path = resolve_path(path)
    files = [
        {"name": item.name, "type": "file" if item.is_file() else "dir"}
        for item in sorted(full_path.iterdir())
    ]
    return {"path": str(full_path), "files": files}


def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """
    Replaces old_str with new_str in a file. If old_str is empty, creates or overwrites the file.
    :param path: The path to the file to edit or create.
    :param old_str: The exact string to replace. Empty string to create a new file.
    :param new_str: The replacement string, or the content of the new file.
    :return: The action taken and the file path.
    """
    full_path = resolve_path(path)
    if old_str == "":
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(new_str, encoding="utf-8")
        return {"path": str(full_path), "action": "created"}
    original = full_path.read_text(encoding="utf-8")
    if old_str not in original:
        return {"path": str(full_path), "action": "old_str_not_found"}
    full_path.write_text(original.replace(old_str, new_str, 1), encoding="utf-8")
    return {"path": str(full_path), "action": "edited"}


TOOLS = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool,
}


# --- System prompt construction ---
def tool_description(name: str) -> str:
    fn = TOOLS[name]
    return f"  Name: {name}\n  Description: {fn.__doc__}\n  Signature: {inspect.signature(fn)}"


def build_system_prompt() -> str:
    tool_repr = "\n\n".join(f"TOOL\n===\n{tool_description(n)}" for n in TOOLS)
    return SYSTEM_PROMPT.format(tool_list_repr=tool_repr)


# --- Tool call parsing ---
def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    calls = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("tool:"):
            continue
        try:
            after = line[5:].strip()
            name, rest = after.split("(", 1)
            name = name.strip()
            if not rest.endswith(")"):
                continue
            args = json.loads(rest[:-1].strip())
            calls.append((name, args))
        except Exception:
            continue
    return calls


# --- LLM call ---
def call_llm(messages: List[Dict]) -> str:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=4096,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks)


# --- Execute a single tool call ---
def run_tool(name: str, args: Dict[str, Any]) -> str:
    print(f"{TOOL_COLOR}tool{RESET}: {name}({json.dumps(args)})")
    fn = TOOLS.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = fn(**args)
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result)


# --- Main agent loop ---
def run():
    system_prompt = build_system_prompt()
    conversation = [{"role": "system", "content": system_prompt}]

    print("Agent ready. Ctrl-C to quit.")
    print(f"Using model: {MODEL}\n")

    while True:
        try:
            user_input = input(f"{YOU_COLOR}You{RESET}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        while True:
            print(f"{ASSISTANT_COLOR}Assistant{RESET}: ", end="", flush=True)
            response = call_llm(conversation)
            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                conversation.append({"role": "assistant", "content": response})
                break

            conversation.append({"role": "assistant", "content": response})

            results = []
            for name, args in tool_calls:
                result = run_tool(name, args)
                results.append(f"tool_result({result})")

            conversation.append({"role": "user", "content": "\n".join(results)})


if __name__ == "__main__":
    run()
