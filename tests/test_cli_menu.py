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
