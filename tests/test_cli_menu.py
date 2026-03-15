from unittest.mock import patch
from chaosengineer.cli_menu import select, _format_options_text


class TestMenuNonInteractive:
    @patch("chaosengineer.cli_menu._is_interactive", return_value=False)
    @patch("builtins.input", return_value="1")
    def test_fallback_returns_first_option(self, mock_input, mock_interactive):
        result = select("Pick one:", ["Alpha", "Beta", "Gamma"])
        assert result == 0

    @patch("chaosengineer.cli_menu._is_interactive", return_value=False)
    @patch("builtins.input", return_value="2")
    def test_fallback_returns_second_option(self, mock_input, mock_interactive):
        result = select("Pick one:", ["Alpha", "Beta", "Gamma"])
        assert result == 1

    @patch("chaosengineer.cli_menu._is_interactive", return_value=False)
    @patch("builtins.input", return_value="invalid")
    def test_fallback_returns_default_on_invalid(self, mock_input, mock_interactive):
        result = select("Pick one:", ["Alpha", "Beta"], default=0)
        assert result == 0


class TestMenuHelpers:
    def test_format_options_non_interactive(self):
        text = _format_options_text(["Alpha", "Beta", "Gamma"])
        assert "1) Alpha" in text
        assert "2) Beta" in text
        assert "3) Gamma" in text


class TestLetterKeyHotkeys:
    def test_letter_key_selects_option(self):
        """Pressing a letter key matching [X] prefix selects that option."""
        from chaosengineer.cli_menu import _match_hotkey
        options = ["[W] Wait for workers", "[K] Kill and pause", "[C] Continue"]
        assert _match_hotkey("w", options) == 0
        assert _match_hotkey("W", options) == 0
        assert _match_hotkey("k", options) == 1
        assert _match_hotkey("K", options) == 1
        assert _match_hotkey("c", options) == 2
        assert _match_hotkey("x", options) is None

    def test_no_hotkey_brackets_returns_none(self):
        from chaosengineer.cli_menu import _match_hotkey
        options = ["Resume previous run", "Start fresh"]
        assert _match_hotkey("r", options) is None
