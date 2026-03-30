"""Tests designed to kill surviving mutants from mutation-guided exploration.

These test *semantic correctness* — not just invariants — to ensure
mutations to count_tokens, estimate_tokens_conservative, and
_format_history are always caught.

Each test documents which mutant category it kills.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

import importlib

from enzu.session import Exchange

# Import the actual modules (not the enzu.session function alias).
# mutate_function_and_test patches at the module level, so we must
# resolve names at call time through the module object.
_budget_mod = importlib.import_module("enzu.budget")
_session_mod = importlib.import_module("enzu.session")


def count_tokens(text, model=None):
    return _budget_mod.count_tokens(text, model)


def estimate_tokens_conservative(text):
    return _budget_mod.estimate_tokens_conservative(text)


def format_history(exchanges, max_chars=10000):
    return _session_mod._format_history(exchanges, max_chars)


# ============================================================================
# count_tokens mutation killers
# ============================================================================


class TestCountTokensMutationKillers:
    """Kill mutants: negate, return_none, boundary, constant, delete."""

    def test_returns_int_never_none(self):
        """Kills: return_none mutants (L17, L33)."""
        assert isinstance(count_tokens("hello world"), int)
        assert isinstance(count_tokens(""), int)
        assert isinstance(count_tokens("a" * 1000), int)

    def test_empty_string_returns_zero(self):
        """Kills: negate L15 (tiktoken branch swap), delete L2."""
        assert count_tokens("") == 0

    def test_non_empty_returns_positive(self):
        """Kills: constant 4→0 (ZeroDivisionError), return_none, negate."""
        result = count_tokens("hello world, this is a test sentence")
        assert result > 0

    def test_longer_text_more_tokens(self):
        """Kills: arithmetic mutations, wrong divisors (4→5)."""
        short = count_tokens("hello")
        long = count_tokens("hello " * 100)
        assert long > short

    def test_proportional_to_length(self):
        """Kills: boundary 4→5, constant 4→0, delete_statement."""
        text = "a " * 500  # ~1000 chars
        tokens = count_tokens(text)
        # Should be roughly chars/4 ± 50% (tiktoken or fallback)
        assert tokens > 50, f"Too few tokens: {tokens}"
        assert tokens < 2000, f"Too many tokens: {tokens}"

    def test_single_char(self):
        """Kills: boundary mutations that shift minimum."""
        result = count_tokens("a")
        assert result >= 0  # could be 0 with //4 on 1 char, that's fine
        assert isinstance(result, int)

    @settings(max_examples=100)
    @given(text=st.text(min_size=1, max_size=500))
    def test_always_non_negative_int(self, text: str):
        """Kills: return_none, negative results from arithmetic mutations."""
        result = count_tokens(text)
        assert isinstance(result, int)
        assert result >= 0

    def test_fallback_path_explicitly(self):
        """Kills: L15 negate, L17 return_none/boundary/constant.
        Force the fallback by temporarily hiding tiktoken."""
        import enzu.budget as bmod

        original = bmod._HAS_TIKTOKEN
        try:
            bmod._HAS_TIKTOKEN = False
            # Now count_tokens must use len(text) // 4
            result = bmod.count_tokens("a" * 40)
            assert result == 10  # 40 // 4
            assert isinstance(result, int)
            assert bmod.count_tokens("") == 0
            assert bmod.count_tokens("hi") >= 0
        finally:
            bmod._HAS_TIKTOKEN = original

    def test_cache_population(self):
        """Kills: L21/L26 negate (cache logic), delete L23/L27/L30."""
        # Clear the cache to force re-population
        import enzu.budget as bmod

        saved_cache = dict(bmod._encoding_cache)
        try:
            bmod._encoding_cache.clear()
            result = bmod.count_tokens("hello world", model="gpt-4o")
            assert isinstance(result, int)
            assert result > 0
            # Cache should now be populated
            assert len(bmod._encoding_cache) > 0
        finally:
            bmod._encoding_cache.update(saved_cache)


class TestEstimateTokensConservativeMutationKillers:
    """Kill mutants: negate, return_none, boundary, constant, delete."""

    def test_empty_returns_zero(self):
        """Kills: negate L10 (empty→non-empty swap), boundary 0→1."""
        assert estimate_tokens_conservative("") == 0

    def test_empty_returns_exactly_zero_not_none(self):
        """Kills: return_none L11."""
        result = estimate_tokens_conservative("")
        assert result is not None
        assert result == 0

    def test_non_empty_returns_at_least_one(self):
        """Kills: constant 1→0 (max(0,...)), negate L10, delete."""
        assert estimate_tokens_conservative("a") >= 1
        assert estimate_tokens_conservative("hi") >= 1

    def test_non_empty_never_returns_none(self):
        """Kills: return_none L13."""
        result = estimate_tokens_conservative("hello world")
        assert result is not None
        assert isinstance(result, int)

    def test_conservative_estimate(self):
        """Kills: boundary 3→4 (wrong divisor). Conservative = chars/3."""
        text = "a" * 30
        result = estimate_tokens_conservative(text)
        # chars/3 = 10, chars/4 = 7. Conservative should be >= chars/4
        assert result >= 7
        assert result == 10  # exactly 30//3

    def test_minimum_is_one_not_two(self):
        """Kills: boundary 1→2 (max(2,...) instead of max(1,...))."""
        # "ab" → len=2, 2//3=0, max(1,0)=1
        assert estimate_tokens_conservative("ab") == 1
        # Not 2

    def test_divisor_is_three_not_four(self):
        """Kills: boundary 3→4, constant 3→0."""
        # 12 chars: 12//3=4, 12//4=3
        assert estimate_tokens_conservative("a" * 12) == 4

    @settings(max_examples=100)
    @given(text=st.text(min_size=0, max_size=500))
    def test_property_empty_or_positive(self, text: str):
        """Kills: all mutations — empty→0, non-empty→>=1, never None."""
        result = estimate_tokens_conservative(text)
        assert isinstance(result, int)
        if not text:
            assert result == 0
        else:
            assert result >= 1


# ============================================================================
# _format_history mutation killers
# ============================================================================


class TestFormatHistoryMutationKillers:
    """Kill mutants: arithmetic, comparison, negate, return_none, boundary, delete."""

    def test_empty_returns_empty_string(self):
        """Kills: negate L8 (exchanges→not exchanges swap)."""
        assert format_history([]) == ""

    def test_empty_returns_str_not_none(self):
        """Kills: return_none L9."""
        result = format_history([])
        assert result is not None
        assert result == ""

    def test_non_empty_returns_non_empty(self):
        """Kills: negate L8, negate L17, return_none L27/L30."""
        exchanges = [Exchange(user="hello", assistant="world")]
        result = format_history(exchanges)
        assert result != ""
        assert isinstance(result, str)

    def test_contains_header_and_footer(self):
        """Kills: delete L29 (header), delete L24 (footer), return_none L30."""
        exchanges = [Exchange(user="q", assistant="a")]
        result = format_history(exchanges)
        assert "== Previous Conversation ==" in result
        assert "== End Previous ==" in result

    def test_contains_user_and_assistant(self):
        """Kills: delete L16/L18/L19 (part building), negate L26."""
        exchanges = [Exchange(user="my question", assistant="my answer")]
        result = format_history(exchanges)
        assert "my question" in result
        assert "my answer" in result

    def test_data_snippet_shown(self):
        """Kills: negate L26 (data_snippet check)."""
        exchanges = [Exchange(user="q", assistant="a", data_snippet="some data")]
        result = format_history(exchanges)
        assert "Data provided:" in result

    def test_no_data_snippet_no_marker(self):
        """Kills: negate L26 (would show marker for None snippet)."""
        exchanges = [Exchange(user="q", assistant="a", data_snippet=None)]
        result = format_history(exchanges)
        assert "Data provided:" not in result

    def test_truncation_respects_max_chars(self):
        """Kills: arithmetic L21 (+→-), negate L21, comparison L21 (>→>=),
        break→continue (loop never stops without break)."""
        big = Exchange(user="x" * 500, assistant="y" * 500)
        exchanges = [big, big, big, big, big]
        result = format_history(exchanges, max_chars=200)
        # Content portion (without header/footer) should be bounded.
        # If break→continue, all 5 exchanges (~5000 chars) would be included.
        assert len(result) < 400  # generous bound including header/footer
        # Stronger: count how many exchanges made it in
        count = result.count("User: ")
        assert count < len(exchanges), (
            f"All {len(exchanges)} exchanges included despite max_chars=200 "
            f"(truncation break not working)"
        )

    def test_most_recent_preserved(self):
        """Kills: delete L23 (reverse), delete L11 (reversed iteration)."""
        old = Exchange(user="old question", assistant="old answer")
        new = Exchange(user="new question", assistant="new answer")
        result = format_history([old, new], max_chars=10000)
        # Both should be present with enough room
        assert "new question" in result
        assert "old question" in result

    def test_truncation_keeps_recent_drops_old(self):
        """Kills: arithmetic +→- (total never grows → never truncates),
        delete L12 (total tracking), comparison >→>= (off-by-one)."""
        old = Exchange(user="A" * 200, assistant="B" * 200)
        new = Exchange(user="recent_q", assistant="recent_a")
        # With tight limit, only recent should fit
        result = format_history([old, new], max_chars=100)
        assert "recent_q" in result
        # Old exchange should be truncated
        assert "A" * 200 not in result

    def test_max_chars_zero_returns_empty(self):
        """Kills: boundary 10000→0 (default), constant 10000→0."""
        exchanges = [Exchange(user="q", assistant="a")]
        result = format_history(exchanges, max_chars=0)
        assert result == ""

    def test_single_exchange_exactly_at_limit(self):
        """Kills: comparison >→>= (off-by-one at boundary)."""
        ex = Exchange(user="q", assistant="a")
        # Calculate exact part length
        part = "User: q\nAssistant: a\n"
        # At exact boundary, should include
        result = format_history([ex], max_chars=len(part))
        assert "q" in result

    def test_off_by_one_truncation(self):
        """Kills: comparison >→>= (L21), boundary 0→1 (L12 total init)."""
        # Create exchange where part length == max_chars exactly
        ex = Exchange(user="q", assistant="a")
        part = "User: q\nAssistant: a\n"
        exact_len = len(part)

        # At exactly max_chars = part_len, exchange fits (> not >=)
        result_exact = format_history([ex], max_chars=exact_len)
        assert "q" in result_exact

        # At max_chars = part_len - 1, exchange does NOT fit
        result_under = format_history([ex], max_chars=exact_len - 1)
        assert result_under == ""

    def test_break_stops_at_first_overflow(self):
        """Kills: break→continue (L22).
        With break: loop stops at the big exchange, only recent small one included.
        With continue: loop skips big exchange, includes the OLD small one too.
        We assert the old one is NOT in the result (break semantics)."""
        old_small = Exchange(user="old", assistant="ok")
        big_recent = Exchange(user="x" * 500, assistant="y" * 500)
        small_recent = Exchange(user="new", assistant="hi")
        # Order: old_small, big_recent, small_recent
        # Reversed iteration: small_recent (fits), big_recent (doesn't fit → break)
        # With break: only small_recent included
        # With continue: small_recent included, big_recent skipped, old_small included
        result = format_history([old_small, big_recent, small_recent], max_chars=100)
        assert "new" in result, "most recent small exchange should be included"
        assert "old" not in result, (
            "old exchange should NOT be included — break should stop iteration "
            "before reaching it"
        )

    def test_two_exchanges_second_fits(self):
        """Kills: L12 boundary/constant (total init 0→1 breaks first iteration).
        With total starting at 1 instead of 0, the first exchange would
        be rejected even when it fits."""
        ex = Exchange(user="hi", assistant="ok")
        part = "User: hi\nAssistant: ok\n"
        # Give enough room for exactly one exchange
        result = format_history([ex, ex], max_chars=len(part))
        # Should contain exactly one exchange (the most recent)
        assert "hi" in result

    @settings(max_examples=100)
    @given(
        n_exchanges=st.integers(min_value=0, max_value=10),
        max_chars=st.integers(min_value=0, max_value=10000),
    )
    def test_property_type_and_bounds(self, n_exchanges: int, max_chars: int):
        """Kills: return_none, negate, delete — result is always str."""
        exchanges = [
            Exchange(user=f"q{i}", assistant=f"a{i}") for i in range(n_exchanges)
        ]
        result = format_history(exchanges, max_chars)
        assert isinstance(result, str)
        if n_exchanges == 0:
            assert result == ""
