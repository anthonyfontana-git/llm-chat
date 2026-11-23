"""llm-chat: a streaming command-line chat client for the Claude API.

Usage:
    python chat.py                         # interactive chat
    python chat.py "Explain recursion"     # one-shot: answer and exit
    python chat.py --system "You are a pirate." --effort low
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per 1M tokens: (input, output). From Anthropic's published pricing.
# Cache writes bill at 1.25x input, cache reads at 0.1x input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens
}


@dataclass
class TurnUsage:
    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int

    @property
    def cost(self) -> float | None:
        """Estimated USD cost of this turn, or None if the model's price is unknown."""
        if self.model not in PRICES:
            return None
        in_price, out_price = PRICES[self.model]
        return (
            self.input_tokens * in_price
            + self.cache_write_tokens * in_price * 1.25
            + self.cache_read_tokens * in_price * 0.10
            + self.output_tokens * out_price
        ) / 1_000_000


class RefusalError(Exception):
    """Claude declined the request (stop_reason == "refusal")."""


class ChatSession:
    """Holds the conversation history and talks to the API.

    The Messages API is stateless, so every request re-sends the whole history.
    Kept separate from the terminal UI so it can be unit-tested with a fake client.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        effort: str = "medium",
        max_tokens: int = 64000,
    ):
        self.client = client
        self.model = model
        self.system = system
        self.effort = effort
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.usage: list[TurnUsage] = []

    def send(self, text: str, on_text: Callable[[str], None] = lambda _: None):
        """Send one user message, streaming the reply's text to `on_text`.

        Returns the final message. On any failure the user message is removed
        again, so the history never ends with an unanswered user turn.
        """
        self.messages.append({"role": "user", "content": text})
        request = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.messages,
            output_config={"effort": self.effort},
            # Cache the conversation prefix so long chats re-bill old turns at 0.1x.
            cache_control={"type": "ephemeral"},
            # If a safety classifier declines, retry on a fallback model server-side.
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if self.system:
            request["system"] = self.system

        try:
            with self.client.beta.messages.stream(**request) as stream:
                for chunk in stream.text_stream:
                    on_text(chunk)
                final = stream.get_final_message()
        except BaseException:  # includes Ctrl-C mid-stream
            self.messages.pop()
            raise

        if final.stop_reason == "refusal":
            self.messages.pop()
            category = final.stop_details.category if final.stop_details else None
            raise RefusalError(category or "unspecified")

        # Store every content block (not just the text): thinking blocks must be
        # sent back unchanged on later turns.
        self.messages.append(
            {"role": "assistant", "content": [b.to_dict(mode="json") for b in final.content]}
        )
        u = final.usage
        self.usage.append(
            TurnUsage(
                model=final.model,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_write_tokens=u.cache_creation_input_tokens or 0,
                cache_read_tokens=u.cache_read_input_tokens or 0,
            )
        )
        return final

    def total_cost(self) -> float:
        return sum(t.cost or 0.0 for t in self.usage)

    def reset(self) -> None:
        self.messages.clear()

    def save(self, path: Path) -> None:
        data = {"model": self.model, "system": self.system, "messages": self.messages}
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.messages = data["messages"]
        self.system = data.get("system", self.system)


# ---------------------------------------------------------------- terminal UI

HELP = """Commands:
  /help            show this help
  /reset           start a new conversation
  /save FILE       save the conversation to a JSON file
  /load FILE       load a saved conversation
  /cost            show token usage and estimated cost
  /quit            exit (Ctrl-D also works)"""


def print_stream(text: str) -> None:
    print(text, end="", flush=True)


def ask(session: ChatSession, text: str) -> bool:
    """Send one message and print the reply. Returns False on a fatal error."""
    try:
        final = session.send(text, on_text=print_stream)
    except KeyboardInterrupt:
        print("\n[interrupted - response discarded]")
        return True
    except RefusalError as e:
        print(f"\n[Claude declined this request (category: {e})]")
        return True
    except TypeError as e:
        if "authentication" not in str(e):
            raise
        print(
            "No API credentials found. Create a key at https://platform.claude.com "
            "and run:\n  export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        return False
    except anthropic.AuthenticationError:
        print("Your API key was rejected - check ANTHROPIC_API_KEY.", file=sys.stderr)
        return False
    except anthropic.RateLimitError as e:
        wait = e.response.headers.get("retry-after", "a few")
        print(f"\n[rate limited - try again in {wait} seconds]")
        return True
    except anthropic.BadRequestError as e:
        print(f"\n[bad request: {e.message}]")
        return True
    except anthropic.APIStatusError as e:
        print(f"\n[API error {e.status_code}: {e.message}]")
        return True
    except anthropic.APIConnectionError:
        print("\n[network error - check your connection]")
        return True

    print()
    if final.stop_reason == "max_tokens":
        print("[reply cut off at the max_tokens limit]")
    if final.model != session.model:
        print(f"[answered by fallback model {final.model}]")
    return True


def show_cost(session: ChatSession) -> None:
    if not session.usage:
        print("No requests yet.")
        return
    tin = sum(t.input_tokens + t.cache_write_tokens + t.cache_read_tokens for t in session.usage)
    tout = sum(t.output_tokens for t in session.usage)
    cached = sum(t.cache_read_tokens for t in session.usage)
    print(
        f"{len(session.usage)} requests | {tin:,} input tokens ({cached:,} from cache) | "
        f"{tout:,} output tokens | ~${session.total_cost():.4f}"
    )


def repl(session: ChatSession) -> None:
    print(f"Chatting with {session.model}. Type /help for commands.")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/reset":
                session.reset()
                print("[new conversation]")
            elif cmd == "/cost":
                show_cost(session)
            elif cmd in ("/save", "/load") and not arg:
                print(f"usage: {cmd} FILE")
            elif cmd == "/save":
                session.save(Path(arg))
                print(f"[saved {len(session.messages)} messages to {arg}]")
            elif cmd == "/load":
                try:
                    session.load(Path(arg))
                    print(f"[loaded {len(session.messages)} messages]")
                except (OSError, json.JSONDecodeError, KeyError) as e:
                    print(f"[could not load {arg}: {e}]")
            else:
                print(f"unknown command {cmd} - try /help")
            continue

        print("\nclaude> ", end="", flush=True)
        if not ask(session, line):
            break
    show_cost(session)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with Claude from the terminal.")
    parser.add_argument("prompt", nargs="?", help="ask one question and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system", help="system prompt (sets Claude's role/behavior)")
    parser.add_argument(
        "--effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="how hard Claude thinks before answering (default: medium)",
    )
    args = parser.parse_args(argv)

    session = ChatSession(
        anthropic.Anthropic(), model=args.model, system=args.system, effort=args.effort
    )
    if args.prompt:
        return 0 if ask(session, args.prompt) else 1
    repl(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
