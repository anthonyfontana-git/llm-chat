# llm-chat

A streaming command-line chat client for the Claude API, built on the official
[`anthropic`](https://pypi.org/project/anthropic/) Python SDK.

```
$ python chat.py --system "You are a concise tutor."
Chatting with claude-opus-5-5. Type /help for commands.

you> What's a hash map?

claude> A hash map stores key-value pairs and uses a hash function ...
```

## Features

- **Streaming output**: text prints token-by-token as the model writes it.
- **Multi-turn memory**: the full conversation is re-sent on each request (the API is stateless).
- **Prompt caching**: the conversation prefix is cached, so earlier turns re-bill at 10% of the input price.
- **Usage and cost tracking**: `/cost` reports input, cached and output tokens plus an estimated dollar cost.
- **Save and resume**: `/save chat.json` and `/load chat.json` store conversations as JSON.
- **Error handling**: separate handling for bad credentials, rate limits (with `retry-after`), bad requests,
  server errors and network failures. A failed or interrupted (Ctrl-C) request is rolled back so the history stays valid.
- **Refusal fallback**: if a safety classifier declines a request, the API retries on a fallback model server-side.
- **Configurable**: `--model`, `--system` prompt, `--effort` (reasoning depth), or one-shot mode (`python chat.py "question"`).
- **Tested offline**: unit tests use a fake client, so they need no API key and cost nothing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
export ANTHROPIC_API_KEY=sk-ant-...   # from https://platform.claude.com
```

## Usage

```bash
python chat.py                                   # interactive chat
python chat.py "Summarize the CAP theorem"       # one question, then exit
python chat.py --effort low --model claude-haiku-5-5   # cheaper and faster
```

In-chat commands: `/help`, `/reset`, `/save FILE`, `/load FILE`, `/cost`, `/quit`.

## Tests

```bash
python -m pytest -q
```

## How it works

`ChatSession` (in `chat.py`) owns the conversation and talks to the API; the terminal UI is a thin
layer on top, which keeps the core logic testable. Each `send()`:

1. appends the user message to the history,
2. opens a streaming request (`client.beta.messages.stream`) with the whole history,
3. forwards text chunks to a callback as they arrive,
4. appends the assistant's full reply (all content blocks, including thinking blocks, which must be
   sent back unchanged) and records token usage from the final message.

If anything fails part-way through, the user message is removed again so the next request is still valid.
