# -*- coding: utf-8 -*-

"""
Unit tests for ModelInfoCache.
Verifies the caching logic for model metadata.
"""

import asyncio
import time
import pytest

from kiro.cache import ModelInfoCache
from kiro.config import DEFAULT_MAX_INPUT_TOKENS


class TestModelInfoCacheInitialization:
    """Tests for ModelInfoCache initialisation."""

    def test_initialization_creates_empty_cache(self):
        """
        What it does: Verifies that the cache is created empty.
        Purpose: Confirm correct initialisation.
        """
        print("Setup: Creating ModelInfoCache...")
        cache = ModelInfoCache()

        print("Check: Cache is empty on creation...")
        print(f"Comparing is_empty(): Expected True, Got {cache.is_empty()}")
        assert cache.is_empty() is True

        print(f"Comparing size: Expected 0, Got {cache.size}")
        assert cache.size == 0

    def test_initialization_with_custom_ttl(self):
        """
        What it does: Verifies cache creation with a custom TTL.
        Purpose: Confirm TTL is configurable.
        """
        print("Setup: Creating ModelInfoCache with TTL=7200...")
        cache = ModelInfoCache(cache_ttl=7200)

        print("Check: TTL is set correctly...")
        print(f"Comparing _cache_ttl: Expected 7200, Got {cache._cache_ttl}")
        assert cache._cache_ttl == 7200

    def test_initialization_last_update_is_none(self):
        """
        What it does: Verifies that last_update_time is initially None.
        Purpose: Confirm the update time is not set before the first update.
        """
        print("Setup: Creating ModelInfoCache...")
        cache = ModelInfoCache()

        print("Check: last_update_time is initially None...")
        print(f"Comparing last_update_time: Expected None, Got {cache.last_update_time}")
        assert cache.last_update_time is None


class TestModelInfoCacheUpdate:
    """Tests for cache updates."""

    @pytest.mark.asyncio
    async def test_update_populates_cache(self, sample_models_data):
        """
        What it does: Verifies the cache is populated with data.
        Purpose: Confirm update() correctly stores models.
        """
        print("Setup: Creating ModelInfoCache...")
        cache = ModelInfoCache()

        print(f"Action: Updating cache with {len(sample_models_data)} models...")
        await cache.update(sample_models_data)

        print("Check: Cache is populated...")
        print(f"Comparing is_empty(): Expected False, Got {cache.is_empty()}")
        assert cache.is_empty() is False

        print(f"Comparing size: Expected {len(sample_models_data)}, Got {cache.size}")
        assert cache.size == len(sample_models_data)

    @pytest.mark.asyncio
    async def test_update_sets_last_update_time(self, sample_models_data):
        """
        What it does: Verifies that the last update time is set.
        Purpose: Confirm last_update_time is set after update.
        """
        print("Setup: Creating ModelInfoCache...")
        cache = ModelInfoCache()

        before_update = time.time()
        print(f"Action: Updating cache (time before: {before_update})...")
        await cache.update(sample_models_data)
        after_update = time.time()

        print("Check: last_update_time is within reasonable bounds...")
        print(f"last_update_time: {cache.last_update_time}")
        assert cache.last_update_time is not None
        assert before_update <= cache.last_update_time <= after_update

    @pytest.mark.asyncio
    async def test_update_replaces_existing_data(self, sample_models_data):
        """
        What it does: Verifies data is replaced on a subsequent update.
        Purpose: Confirm old data is fully replaced.
        """
        print("Setup: Creating ModelInfoCache and performing first update...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Action: Updating with new data...")
        new_data = [{"modelId": "new-model", "tokenLimits": {"maxInputTokens": 50000}}]
        await cache.update(new_data)

        print("Check: Old data is replaced...")
        print(f"Comparing size: Expected 1, Got {cache.size}")
        assert cache.size == 1

        print("Check: Old model is unavailable...")
        assert cache.get("claude-sonnet-4") is None

        print("Check: New model is available...")
        assert cache.get("new-model") is not None

    @pytest.mark.asyncio
    async def test_update_with_empty_list(self):
        """
        What it does: Verifies update with an empty list.
        Purpose: Confirm the cache is cleared on an empty update.
        """
        print("Setup: Creating ModelInfoCache with data...")
        cache = ModelInfoCache()
        await cache.update([{"modelId": "test-model"}])

        print("Action: Updating with an empty list...")
        await cache.update([])

        print("Check: Cache is empty...")
        print(f"Comparing is_empty(): Expected True, Got {cache.is_empty()}")
        assert cache.is_empty() is True


class TestModelInfoCacheGet:
    """Tests for retrieving data from the cache."""

    @pytest.mark.asyncio
    async def test_get_returns_model_info(self, sample_models_data):
        """
        What it does: Verifies retrieval of model information.
        Purpose: Confirm get() returns correct data.
        """
        print("Setup: Creating and populating cache...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Action: Retrieving info for claude-sonnet-4...")
        model_info = cache.get("claude-sonnet-4")

        print("Check: Information retrieved...")
        print(f"model_info: {model_info}")
        assert model_info is not None
        assert model_info["modelId"] == "claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown_model(self, sample_models_data):
        """
        What it does: Verifies None is returned for an unknown model.
        Purpose: Confirm get() does not fail when the model is absent.
        """
        print("Setup: Creating and populating cache...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Action: Retrieving info for a non-existent model...")
        model_info = cache.get("non-existent-model")

        print("Check: None was returned...")
        print(f"Comparing model_info: Expected None, Got {model_info}")
        assert model_info is None

    def test_get_from_empty_cache(self):
        """
        What it does: Verifies get() on an empty cache.
        Purpose: Confirm an empty cache does not raise errors.
        """
        print("Setup: Creating an empty cache...")
        cache = ModelInfoCache()

        print("Action: Retrieving from an empty cache...")
        model_info = cache.get("any-model")

        print("Check: None was returned...")
        print(f"Comparing model_info: Expected None, Got {model_info}")
        assert model_info is None


class TestModelInfoCacheGetMaxInputTokens:
    """Tests for retrieving maxInputTokens."""

    @pytest.mark.asyncio
    async def test_get_max_input_tokens_returns_value(self, sample_models_data):
        """
        What it does: Verifies retrieval of maxInputTokens for a model.
        Purpose: Confirm the value is extracted from tokenLimits.
        """
        print("Setup: Creating and populating cache...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Action: Retrieving maxInputTokens for claude-sonnet-4...")
        max_tokens = cache.get_max_input_tokens("claude-sonnet-4")

        print("Check: Value is correct...")
        print(f"Comparing max_tokens: Expected 200000, Got {max_tokens}")
        assert max_tokens == 200000

    @pytest.mark.asyncio
    async def test_get_max_input_tokens_returns_default_for_unknown(self, sample_models_data):
        """
        What it does: Verifies default return for an unknown model.
        Purpose: Confirm DEFAULT_MAX_INPUT_TOKENS is returned.
        """
        print("Setup: Creating and populating cache...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Action: Retrieving maxInputTokens for an unknown model...")
        max_tokens = cache.get_max_input_tokens("unknown-model")

        print("Check: Default was returned...")
        print(f"Comparing max_tokens: Expected {DEFAULT_MAX_INPUT_TOKENS}, Got {max_tokens}")
        assert max_tokens == DEFAULT_MAX_INPUT_TOKENS

    @pytest.mark.asyncio
    async def test_get_max_input_tokens_returns_default_when_no_token_limits(self):
        """
        What it does: Verifies default return when tokenLimits is absent.
        Purpose: Confirm a model without tokenLimits does not break the logic.
        """
        print("Setup: Creating cache with a model without tokenLimits...")
        cache = ModelInfoCache()
        await cache.update([{"modelId": "model-without-limits"}])

        print("Action: Retrieving maxInputTokens...")
        max_tokens = cache.get_max_input_tokens("model-without-limits")

        print("Check: Default was returned...")
        print(f"Comparing max_tokens: Expected {DEFAULT_MAX_INPUT_TOKENS}, Got {max_tokens}")
        assert max_tokens == DEFAULT_MAX_INPUT_TOKENS

    @pytest.mark.asyncio
    async def test_get_max_input_tokens_returns_default_when_max_input_is_none(self):
        """
        What it does: Verifies default return when maxInputTokens is None.
        Purpose: Confirm None in tokenLimits is handled correctly.
        """
        print("Setup: Creating cache with a model with maxInputTokens=None...")
        cache = ModelInfoCache()
        await cache.update([{
            "modelId": "model-with-null",
            "tokenLimits": {"maxInputTokens": None}
        }])

        print("Action: Retrieving maxInputTokens...")
        max_tokens = cache.get_max_input_tokens("model-with-null")

        print("Check: Default was returned...")
        print(f"Comparing max_tokens: Expected {DEFAULT_MAX_INPUT_TOKENS}, Got {max_tokens}")
        assert max_tokens == DEFAULT_MAX_INPUT_TOKENS


class TestModelInfoCacheIsEmpty:
    """Tests for checking whether the cache is empty."""

    def test_is_empty_returns_true_for_new_cache(self):
        """
        What it does: Verifies is_empty() for a new cache.
        Purpose: Confirm a new cache is considered empty.
        """
        print("Setup: Creating a new cache...")
        cache = ModelInfoCache()

        print("Check: is_empty() returns True...")
        print(f"Comparing is_empty(): Expected True, Got {cache.is_empty()}")
        assert cache.is_empty() is True

    @pytest.mark.asyncio
    async def test_is_empty_returns_false_after_update(self, sample_models_data):
        """
        What it does: Verifies is_empty() after population.
        Purpose: Confirm a populated cache is not considered empty.
        """
        print("Setup: Creating and populating cache...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Check: is_empty() returns False...")
        print(f"Comparing is_empty(): Expected False, Got {cache.is_empty()}")
        assert cache.is_empty() is False


class TestModelInfoCacheIsStale:
    """Tests for checking whether the cache is stale."""

    def test_is_stale_returns_true_for_new_cache(self):
        """
        What it does: Verifies is_stale() for a new cache.
        Purpose: Confirm a cache without updates is considered stale.
        """
        print("Setup: Creating a new cache...")
        cache = ModelInfoCache()

        print("Check: is_stale() returns True...")
        print(f"Comparing is_stale(): Expected True, Got {cache.is_stale()}")
        assert cache.is_stale() is True

    @pytest.mark.asyncio
    async def test_is_stale_returns_false_after_recent_update(self, sample_models_data):
        """
        What it does: Verifies is_stale() immediately after an update.
        Purpose: Confirm a fresh cache is not considered stale.
        """
        print("Setup: Creating and populating cache...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Check: is_stale() returns False...")
        print(f"Comparing is_stale(): Expected False, Got {cache.is_stale()}")
        assert cache.is_stale() is False

    @pytest.mark.asyncio
    async def test_is_stale_returns_true_after_ttl_expires(self, sample_models_data):
        """
        What it does: Verifies is_stale() after TTL expiry.
        Purpose: Confirm the cache is considered stale after TTL.
        """
        print("Setup: Creating cache with TTL=0.1 seconds...")
        cache = ModelInfoCache(cache_ttl=0.1)
        await cache.update(sample_models_data)

        print("Action: Waiting for TTL to expire...")
        await asyncio.sleep(0.2)

        print("Check: is_stale() returns True...")
        print(f"Comparing is_stale(): Expected True, Got {cache.is_stale()}")
        assert cache.is_stale() is True


class TestModelInfoCacheGetAllModelIds:
    """Tests for retrieving the list of model IDs."""

    def test_get_all_model_ids_returns_empty_for_new_cache(self):
        """
        What it does: Verifies get_all_model_ids() for an empty cache.
        Purpose: Confirm an empty list is returned.
        """
        print("Setup: Creating an empty cache...")
        cache = ModelInfoCache()

        print("Action: Retrieving the list of model IDs...")
        model_ids = cache.get_all_model_ids()

        print("Check: List is empty...")
        print(f"Comparing model_ids: Expected [], Got {model_ids}")
        assert model_ids == []

    @pytest.mark.asyncio
    async def test_get_all_model_ids_returns_all_ids(self, sample_models_data):
        """
        What it does: Verifies get_all_model_ids() for a populated cache.
        Purpose: Confirm all model IDs are returned.
        """
        print("Setup: Creating and populating cache...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Action: Retrieving the list of model IDs...")
        model_ids = cache.get_all_model_ids()

        print("Check: All IDs are present...")
        expected_ids = [m["modelId"] for m in sample_models_data]
        print(f"Comparing model_ids: Expected {expected_ids}, Got {model_ids}")
        assert set(model_ids) == set(expected_ids)


class TestModelInfoCacheThreadSafety:
    """Tests for cache thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_updates_dont_corrupt_cache(self, sample_models_data):
        """
        What it does: Verifies thread safety under concurrent updates.
        Purpose: Confirm asyncio.Lock protects against race conditions.
        """
        print("Setup: Creating cache...")
        cache = ModelInfoCache()

        async def update_with_data(data):
            await cache.update(data)

        print("Action: 10 concurrent updates...")
        tasks = []
        for i in range(10):
            data = [{"modelId": f"model-{i}", "tokenLimits": {"maxInputTokens": 100000 + i}}]
            tasks.append(update_with_data(data))

        await asyncio.gather(*tasks)

        print("Check: Cache contains the data from the last update...")
        # Due to race conditions, we cannot know which update was last,
        # but the cache must contain exactly one model
        print(f"Comparing size: Expected 1, Got {cache.size}")
        assert cache.size == 1

        print("Check: Cache is not corrupted...")
        model_ids = cache.get_all_model_ids()
        assert len(model_ids) == 1
        assert model_ids[0].startswith("model-")

    @pytest.mark.asyncio
    async def test_concurrent_reads_are_safe(self, sample_models_data):
        """
        What it does: Verifies safety of concurrent reads.
        Purpose: Confirm multiple get() calls do not cause issues.
        """
        print("Setup: Creating and populating cache...")
        cache = ModelInfoCache()
        await cache.update(sample_models_data)

        print("Action: 100 concurrent reads...")
        async def read_model():
            return cache.get("claude-sonnet-4")

        results = await asyncio.gather(*[read_model() for _ in range(100)])

        print("Check: All reads returned the same result...")
        assert all(r is not None for r in results)
        assert all(r["modelId"] == "claude-sonnet-4" for r in results)
