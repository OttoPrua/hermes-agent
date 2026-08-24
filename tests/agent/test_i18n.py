"""Tests for agent.i18n -- catalog parity, fallback, language resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent import i18n


LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"


def _load_raw(lang: str) -> dict:
    with (LOCALES_DIR / f"{lang}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d, prefix="") -> dict:
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------
# Catalog completeness -- this is the key invariant test.  If someone adds a
# new key to en.yaml they MUST add it to every other locale, else runtime
# falls back to English for those users and defeats the feature.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lang", [l for l in i18n.SUPPORTED_LANGUAGES if l != "en"])
def test_catalog_keys_match_english(lang: str):
    """Every non-English catalog must have exactly the same key set as English."""
    en_keys = set(_flatten(_load_raw("en")).keys())
    lang_keys = set(_flatten(_load_raw(lang)).keys())
    missing = en_keys - lang_keys
    extra = lang_keys - en_keys
    assert not missing, f"{lang}.yaml missing keys: {sorted(missing)}"
    assert not extra, f"{lang}.yaml has keys not in en.yaml: {sorted(extra)}"


@pytest.mark.parametrize("lang", list(i18n.SUPPORTED_LANGUAGES))
def test_catalog_placeholders_match_english(lang: str):
    """Every translated value must use the same {placeholder} tokens as English.

    A mistranslated placeholder (e.g. ``{description}`` typoed as ``{descricao}``)
    would either raise KeyError at runtime or silently drop the interpolated
    value.  Pin parity at the test layer.
    """
    import re

    placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    en_flat = _flatten(_load_raw("en"))
    lang_flat = _flatten(_load_raw(lang))
    for key, en_value in en_flat.items():
        en_placeholders = set(placeholder_re.findall(en_value))
        lang_value = lang_flat.get(key, "")
        lang_placeholders = set(placeholder_re.findall(lang_value))
        assert en_placeholders == lang_placeholders, (
            f"{lang}.yaml key={key!r}: placeholders {lang_placeholders} "
            f"don't match English {en_placeholders}"
        )


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------


def test_default_when_nothing_set(monkeypatch):
    """With no env var and no config override, falls back to English."""
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    # Force config lookup to return None -- patch the cached reader.
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    assert i18n.get_language() == "en"


# ---------------------------------------------------------------------------
# t() semantics
# ---------------------------------------------------------------------------


def test_t_missing_key_in_non_english_falls_back_to_english(tmp_path, monkeypatch):
    """If a key exists in English but not in the target locale, fall back."""
    # Stand up a fake incomplete locale under a temp locales dir.
    fake_locales = tmp_path / "locales"
    fake_locales.mkdir()
    (fake_locales / "en.yaml").write_text("foo: English Foo\n", encoding="utf-8")
    (fake_locales / "zh.yaml").write_text("# intentionally empty\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake_locales)
    i18n.reset_language_cache()
    try:
        assert i18n.t("foo", lang="zh") == "English Foo"
    finally:
        # Clear the cache on teardown so subsequent tests don't see the
        # fake "foo: English Foo" catalog instead of the real locales/*.yaml.
        i18n.reset_language_cache()


# ---------------------------------------------------------------------------
# _locales_dir resolution ladder -- regression for #23943 / #27632 / #35374.
# Sealed installs (Nix store venv, pip wheel) have no source tree next to
# agent/, so _locales_dir must resolve via env override or the data scheme.
# ---------------------------------------------------------------------------


def test_locales_dir_env_override_ignored_when_missing(tmp_path, monkeypatch):
    """A bogus HERMES_BUNDLED_LOCALES falls through to source/wheel resolution
    instead of returning a path that doesn't exist."""
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", str(tmp_path / "does-not-exist"))
    result = i18n._locales_dir()
    assert result != tmp_path / "does-not-exist"
    # In a source checkout this is the repo-root locales dir.
    assert result.name == "locales"


# ---------------------------------------------------------------------------
# #90662 — runtime status / busy-wait keys (thin-slice expansion)
# ---------------------------------------------------------------------------


_RUNTIME_KEYS = (
    "agent.runtime.starting_api_call",
    "agent.runtime.api_error_recovery",
    "agent.runtime.no_fallback_provider",
    "agent.runtime.empty_malformed_fallback",
    "agent.runtime.safety_refusal_fallback",
    "agent.runtime.content_filter_fallback",
    "agent.runtime.billing_exhausted_fallback",
    "agent.runtime.rate_limited_exhausted",
    "agent.runtime.billing_terminal_label",
    "gateway.runtime.subagent_working",
    "gateway.runtime.compressing_context",
    "gateway.runtime.queued_next_turn",
    "gateway.runtime.interrupting",
    "gateway.runtime.queued_short",
    "gateway.runtime.steer_queued_starting",
    "gateway.runtime.drain_queued_restart",
    "gateway.runtime.drain_not_accepting_work_shutdown",
    "gateway.runtime.confirm_new_detail",
    "gateway.runtime.confirm_undo_one",
)


def test_runtime_status_keys_exist_and_zh_is_translated():
    """#90662 keys must resolve in English and differ under display.language=zh."""
    i18n.reset_language_cache()
    for key in _RUNTIME_KEYS:
        en = i18n.t(
            key,
            lang="en",
            n=3,
            attempt=1,
            max=2,
            summary="boom",
            detail="",
            depth=2,
            count=3,
            provider="openai",
            status_code=400,
        )
        zh = i18n.t(
            key,
            lang="zh",
            n=3,
            attempt=1,
            max=2,
            summary="boom",
            detail="",
            depth=2,
            count=3,
            provider="openai",
            status_code=400,
        )
        assert en and not en.startswith(key), f"missing English catalog entry {key}"
        assert zh and not zh.startswith(key), f"missing zh catalog entry {key}"
        assert zh != en, f"{key} zh translation fell back to English"


def test_billing_terminal_label_honors_language(monkeypatch):
    from agent.conversation_loop import _billing_terminal_label

    i18n.reset_language_cache()
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    assert (
        _billing_terminal_label("boom", False) == "Billing or credits exhausted: boom"
    )

    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    i18n.reset_language_cache()
    try:
        assert _billing_terminal_label("boom", False) == "账单或额度已耗尽：boom"
    finally:
        monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
        i18n.reset_language_cache()


def test_noisy_status_regex_still_matches_english_compression_and_max_retries():
    """Compression-progress and max-retries chatter must stay English-matchable.

    Translating those strings would leak them into Telegram/Slack because
    ``_TELEGRAM_NOISY_STATUS_RE`` is English-only. #90662 leaves them alone.
    """
    from agent.conversation_compression import ROUTINE_COMPRESSION_STATUS_SAMPLES
    from gateway.run import _TELEGRAM_NOISY_STATUS_RE

    for message in ROUTINE_COMPRESSION_STATUS_SAMPLES:
        assert _TELEGRAM_NOISY_STATUS_RE.search(message), message

    leftover_english = (
        "⚠️ Max retries (3) for invalid responses — trying fallback...",
        "⚠️ Max retries (3) exhausted — trying fallback...",
        "❌ Max retries (3) exceeded for invalid responses. Giving up.",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        "🗜️ Preflight compression check before sending...",
        (
            "📦 Pre-API compression: ~123,456 tokens near the context/output limit. "
            "Compacting before the next model call."
        ),
    )
    for message in leftover_english:
        assert _TELEGRAM_NOISY_STATUS_RE.search(message), message
