"""Interactive arrow-key menu for CLI prompts."""
from __future__ import annotations
import re
import sys


def _match_hotkey(ch: str, options: list[str]) -> int | None:
    """Match a single character against [X] hotkey prefixes in options."""
    for i, opt in enumerate(options):
        m = re.match(r"\[(\w)\]", opt)
        if m and m.group(1).lower() == ch.lower():
            return i
    return None


def _is_interactive() -> bool:
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _format_options_text(options: list[str]) -> str:
    return "\n".join(f"  {i + 1}) {opt}" for i, opt in enumerate(options))


def select(prompt: str, options: list[str], default: int = 0) -> int:
    """Show interactive menu, return selected index."""
    if not _is_interactive():
        return _select_text(prompt, options, default)
    return _select_interactive(prompt, options, default)


def _select_text(prompt: str, options: list[str], default: int) -> int:
    """Non-interactive fallback: numbered list with text input."""
    print(f"\n{prompt}\n")
    print(_format_options_text(options))
    try:
        choice = input(f"\nEnter choice [1-{len(options)}]: ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return idx
    except (ValueError, EOFError):
        pass
    return default


def _select_interactive(prompt: str, options: list[str], default: int) -> int:
    """Interactive mode: arrow keys + Enter."""
    import tty
    import termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    selected = default
    try:
        tty.setraw(fd)
        _render(prompt, options, selected, first=True)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\r" or ch == "\n":
                break
            elif ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    selected = (selected - 1) % len(options)
                elif seq == "[B":
                    selected = (selected + 1) % len(options)
                _render(prompt, options, selected, first=False)
            elif ch == "\x03":
                raise KeyboardInterrupt
            else:
                match = _match_hotkey(ch, options)
                if match is not None:
                    selected = match
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return selected


def _render(prompt: str, options: list[str], selected: int, first: bool = False) -> None:
    lines = 2 + len(options)
    if not first:
        sys.stdout.write(f"\r\033[{lines}A\033[J")
    sys.stdout.write(f"\n{prompt}\n\n")
    for i, opt in enumerate(options):
        marker = "\033[1m  → \033[0m" if i == selected else "    "
        sys.stdout.write(f"{marker}{opt}\n")
    sys.stdout.flush()
