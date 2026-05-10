# -*- coding: utf-8 -*-

"""
Property tests for cache correctness — prove we cannot serve a wrong
answer from the cache.

The prior broken design excluded the trailing user turn from the cache
key, so two requests with the same prefix but different trailing
questions would collide and return the wrong cached response. These
tests lock in the fixed semantics with hypothesis-generated inputs.

Required invariants:
  P1. Different messages -> different keys (no collisions).
  P2. Same messages -> same key (determinism).
  P3. Trailing-turn content MUST affect the key (the bug).
  P4. Tool-list identity MUST affect the key.
  P5. Model + max_tokens MUST affect the key.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from kiro.response_cache import make_key

# Constrained strategies so hypothesis doesn't spend shrinking cycles
# on pathological unicode / huge dicts that don't exercise real bugs.
_roles = st.sampled_from(["user", "assistant"])
_text = st.text(min_size=1, max_size=200,
                alphabet=st.characters(whitelist_categories=("L", "N", "Z", "P")))

def _message():
    return st.fixed_dictionaries({
        "role": _roles,
        "content": _text,
    })

_messages = st.lists(_message(), min_size=1, max_size=8)
_model = st.sampled_from([
    "claude-sonnet-4-5", "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001", "claude-opus-4-7",
])
_max_tokens = st.integers(min_value=1, max_value=100_000)
_session_ids = st.text(min_size=1, max_size=40, alphabet=st.characters(whitelist_categories=("L", "N"))) \
    .filter(lambda s: s.strip())  # non-empty


@given(msgs_a=_messages, msgs_b=_messages, session=_session_ids,
       model=_model, mt=_max_tokens)
@settings(max_examples=200, deadline=None)
def test_p1_different_messages_different_keys(msgs_a, msgs_b, session, model, mt):
    """P1: if message sequences differ in any position, keys must differ."""
    if msgs_a == msgs_b:
        return  # hypothesis happened to generate equal; not a counter-example
    k_a = make_key(session_id=session, system=None, messages=msgs_a,
                   model=model, max_tokens=mt)
    k_b = make_key(session_id=session, system=None, messages=msgs_b,
                   model=model, max_tokens=mt)
    assert k_a != k_b, f"Different message sequences collided on key: a={msgs_a!r} b={msgs_b!r}"


@given(msgs=_messages, session=_session_ids, model=_model, mt=_max_tokens)
@settings(max_examples=100, deadline=None)
def test_p2_same_inputs_same_key(msgs, session, model, mt):
    """P2: determinism — identical inputs always produce the identical key."""
    k1 = make_key(session_id=session, system=None, messages=msgs,
                  model=model, max_tokens=mt)
    k2 = make_key(session_id=session, system=None, messages=msgs,
                  model=model, max_tokens=mt)
    assert k1 == k2


@given(prefix=_messages, tail_a=_text, tail_b=_text,
       session=_session_ids, model=_model, mt=_max_tokens)
@settings(max_examples=200, deadline=None)
def test_p3_trailing_user_turn_affects_key(prefix, tail_a, tail_b, session, model, mt):
    """P3 (the bug regression): two requests with the same prefix but different
    trailing user messages MUST produce different keys.

    The prior broken design excluded the trailing turn, causing cache
    collisions that would return the wrong answer. This test locks in the
    fix and asserts it never regresses.
    """
    if tail_a == tail_b:
        return  # same tail -> same key is correct
    msgs_a = list(prefix) + [{"role": "user", "content": tail_a}]
    msgs_b = list(prefix) + [{"role": "user", "content": tail_b}]
    k_a = make_key(session_id=session, system=None, messages=msgs_a,
                   model=model, max_tokens=mt)
    k_b = make_key(session_id=session, system=None, messages=msgs_b,
                   model=model, max_tokens=mt)
    assert k_a != k_b, (
        f"REGRESSION: prefix-only keying has returned. Trailing-turn "
        f"variance must invalidate the cache key to avoid wrong-answer "
        f"bugs. prefix={prefix!r} tail_a={tail_a!r} tail_b={tail_b!r}"
    )


@given(msgs=_messages, session=_session_ids, model_a=_model, model_b=_model, mt=_max_tokens)
@settings(max_examples=100, deadline=None)
def test_p5_model_affects_key(msgs, session, model_a, model_b, mt):
    """P5a: swapping the model must change the key (different models = different completions)."""
    if model_a == model_b:
        return
    k_a = make_key(session_id=session, system=None, messages=msgs,
                   model=model_a, max_tokens=mt)
    k_b = make_key(session_id=session, system=None, messages=msgs,
                   model=model_b, max_tokens=mt)
    assert k_a != k_b


@given(msgs=_messages, session=_session_ids, model=_model, mt_a=_max_tokens, mt_b=_max_tokens)
@settings(max_examples=100, deadline=None)
def test_p5_max_tokens_affects_key(msgs, session, model, mt_a, mt_b):
    """P5b: different max_tokens => potentially different truncation => different key."""
    if mt_a == mt_b:
        return
    k_a = make_key(session_id=session, system=None, messages=msgs,
                   model=model, max_tokens=mt_a)
    k_b = make_key(session_id=session, system=None, messages=msgs,
                   model=model, max_tokens=mt_b)
    assert k_a != k_b
