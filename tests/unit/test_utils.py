"""Unit tests for kiro/utils.py."""
from __future__ import annotations

import uuid

from kiro.utils import (
    generate_completion_id,
    generate_conversation_id,
    generate_tool_call_id,
    get_machine_fingerprint,
)


class TestGetMachineFingerprint:
    def test_returns_64_char_hex(self):
        fp = get_machine_fingerprint()
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_deterministic(self):
        fp1 = get_machine_fingerprint()
        fp2 = get_machine_fingerprint()
        assert fp1 == fp2


class TestGenerateCompletionId:
    def test_format(self):
        cid = generate_completion_id()
        assert cid.startswith("chatcmpl-")
        assert len(cid) > 10

    def test_unique(self):
        ids = {generate_completion_id() for _ in range(10)}
        assert len(ids) == 10


class TestGenerateConversationId:
    def test_no_messages_returns_uuid(self):
        cid = generate_conversation_id()
        # Should be a valid UUID string
        uuid.UUID(cid)

    def test_empty_list_returns_uuid(self):
        cid = generate_conversation_id([])
        uuid.UUID(cid)

    def test_with_messages_returns_16_char_hex(self):
        messages = [{"role": "user", "content": "Hello"}]
        cid = generate_conversation_id(messages)
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)

    def test_same_messages_same_id(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        id1 = generate_conversation_id(messages)
        id2 = generate_conversation_id(messages)
        assert id1 == id2

    def test_different_messages_different_id(self):
        msgs1 = [{"role": "user", "content": "Hello"}]
        msgs2 = [{"role": "user", "content": "Goodbye"}]
        assert generate_conversation_id(msgs1) != generate_conversation_id(msgs2)

    def test_stable_across_conversation_growth(self):
        """Adding messages beyond the first 3 doesn't change the ID."""
        base = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        extended = base + [{"role": "assistant", "content": "Fine thanks"}]
        assert generate_conversation_id(base) == generate_conversation_id(extended)

    def test_list_content_handled(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        cid = generate_conversation_id(messages)
        assert len(cid) == 16

    def test_dict_content_handled(self):
        messages = [{"role": "user", "content": {"type": "text", "text": "Hello"}}]
        cid = generate_conversation_id(messages)
        assert len(cid) == 16


class TestGenerateToolCallId:
    def test_format(self):
        tid = generate_tool_call_id()
        assert tid.startswith("call_")
        assert len(tid) == 13  # "call_" + 8 hex chars

    def test_unique(self):
        ids = {generate_tool_call_id() for _ in range(10)}
        assert len(ids) == 10
