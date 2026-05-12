# harness

A minimal Python agent harness using OpenRouter.

## Setup

```bash
pip install openai python-dotenv
```

Add your key to `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

Run:
```bash
python3 agent.py
```

## Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read the contents of a file |
| `list_files` | List files and directories at a path |
| `edit_file` | Create or edit a file via string replacement |
| `bash` | Run a shell command and return stdout/stderr/exit code |

## Commands

| Command | Description |
|---------|-------------|
| `/new` | Clear history and start a fresh session |
| `/exit` | Save session and quit |
| `Ctrl-C` | Quit (session is still resumable) |

## Session persistence

Conversation history is saved to `.agent_history.json` in the current directory and automatically resumed on next run.
