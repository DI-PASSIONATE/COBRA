"""Token tables and stylesheet of the GUI theme; runs without a display.

PySide6 is imported inside a fixture rather than at module level so that
collecting this file does not pull Qt into ``sys.modules`` for the CLI tests.
"""

import re
from dataclasses import fields

import pytest

HEX_RE = re.compile(r"^#[0-9A-F]{6}$")
THEME_NAMES = ("SANDBANK", "DEEPWATER")


@pytest.fixture(scope="module")
def theme():
    pytest.importorskip("PySide6")
    from cobra.gui import theme

    return theme


@pytest.fixture(params=THEME_NAMES)
def tokens(theme, request):
    return getattr(theme, request.param)


def _luminance(hex_color: str) -> float:
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_every_colour_token_is_a_hex_literal(tokens):
    for field in fields(tokens):
        value = getattr(tokens, field.name)
        if field.name == "series":
            assert all(HEX_RE.match(c) for c in value)
        elif isinstance(value, str) and field.name != "name":
            assert HEX_RE.match(value), f"{field.name} = {value!r}"


def test_themes_are_structurally_identical(theme):
    assert {f.name for f in fields(theme.SANDBANK)} == {f.name for f in fields(theme.DEEPWATER)}
    assert len(theme.SANDBANK.series) == len(theme.DEEPWATER.series) == 8
    assert theme.SANDBANK.dark is False
    assert theme.DEEPWATER.dark is True


def test_contrast_targets(tokens):
    assert contrast(tokens.text, tokens.surface) >= 7
    assert contrast(tokens.text, tokens.canvas) >= 7
    assert contrast(tokens.text_muted, tokens.surface) >= 4.5
    assert contrast(tokens.on_tide, tokens.tide) >= 4.5
    for strong in (tokens.moss_strong, tokens.ochre_strong, tokens.ember_strong):
        assert contrast(strong, tokens.surface) >= 4.5
        assert contrast(tokens.on_tide, strong) >= 4.5
    assert contrast(tokens.tide, tokens.surface) >= 3
    assert contrast(tokens.border, tokens.surface) >= 1.3


def test_stylesheet_substitutes_every_placeholder(theme, tokens):
    stylesheet = theme.build_stylesheet(tokens)
    assert "$" not in stylesheet
    assert tokens.canvas in stylesheet
    assert tokens.tide in stylesheet
    for state in ("pause", "resume", "stopping"):
        assert f'actionState="{state}"' in stylesheet


def test_stylesheet_embeds_the_check_icon(theme, tokens, tmp_path):
    check = tmp_path / "check.svg"
    assert "image: url(" not in theme.build_stylesheet(tokens)
    assert f"image: url({check.as_posix()});" in theme.build_stylesheet(tokens, check)


def test_series_colours_cycle(theme):
    tokens = theme.SANDBANK
    assert tokens.series_color(0) == tokens.series[0]
    assert tokens.series_color(len(tokens.series)) == tokens.series[0]
