"""Unit tests for ChatSession. A fake client stands in for the API, so these
run offline, cost nothing, and need no API key."""

import pytest
from anthropic.types.beta import BetaMessage

from chat import ChatSession, RefusalError, TurnUsage


def make_message(text="Hello!", stop_reason="end_turn", model="claude-opus-5-5", **extra):
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig123"},
                {"type": "text", "text": text},
            ],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 2000,
            },
            **extra,
        }
    )


class FakeStream:
    def __init__(self, final, chunks=("Hel", "lo!"), error=None):
        self.final, self.text_stream, self.error = final, self._gen(chunks, error), error

    @staticmethod
    def _gen(chunks, error):
        yield from chunks
        if error:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.final


class FakeClient:
    """Mimics client.beta.messages.stream(...) and records each request."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self.requests = []
        self.beta = self
        self.messages = self

    def stream(self, **request):
        # Snapshot the history: the session list keeps changing after the call.
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.streams.pop(0)


def test_send_streams_text_and_records_history():
    client = FakeClient(FakeStream(make_message()))
    session = ChatSession(client, system="Be brief.")
    chunks = []

    session.send("Hi", on_text=chunks.append)

    assert "".join(chunks) == "Hello!"
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    # Thinking blocks are stored too, so they can be sent back next turn.
    assert session.messages[1]["content"][0]["type"] == "thinking"
    req = client.requests[0]
    assert req["system"] == "Be brief."
    assert req["fallbacks"] == "default"
    assert req["output_config"] == {"effort": "medium"}


def test_second_turn_resends_full_history():
    client = FakeClient(FakeStream(make_message("A")), FakeStream(make_message("B")))
    session = ChatSession(client)
    session.send("first")
    session.send("second")
    assert [m["role"] for m in client.requests[1]["messages"]] == ["user", "assistant", "user"]


def test_no_system_key_when_unset():
    client = FakeClient(FakeStream(make_message()))
    ChatSession(client).send("Hi")
    assert "system" not in client.requests[0]


def test_failed_request_rolls_back_user_message():
    client = FakeClient(FakeStream(make_message(), error=ConnectionError("dropped")))
    session = ChatSession(client)
    with pytest.raises(ConnectionError):
        session.send("Hi")
    assert session.messages == []


def test_refusal_rolls_back_and_raises():
    final = make_message(
        "", stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    session = ChatSession(FakeClient(FakeStream(final, chunks=())))
    with pytest.raises(RefusalError, match="cyber"):
        session.send("something sketchy")
    assert session.messages == []


def test_cost_math():
    # Opus 5.5: $4/M input, $20/M output, cache read at 10% of input.
    turn = TurnUsage("claude-opus-5-5", 1000, 500, 0, 2000)
    assert turn.cost == pytest.approx((1000 * 4 + 2000 * 0.4 + 500 * 20) / 1e6)
    assert TurnUsage("some-unknown-model", 1, 1, 0, 0).cost is None


def test_save_and_load_roundtrip(tmp_path):
    session = ChatSession(FakeClient(FakeStream(make_message())), system="sys")
    session.send("Hi")
    path = tmp_path / "convo.json"
    session.save(path)

    restored = ChatSession(FakeClient())
    restored.load(path)
    assert restored.messages == session.messages
    assert restored.system == "sys"
