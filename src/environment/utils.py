"""
Utility functions for Lean 4 environment.
Includes parsing, formatting, and validation helpers.
"""
from typing import Optional, Dict, Any
import re
import sys


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal formatting."""

    # Reset
    RESET = "\033[0m"

    # Regular colors
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"

    # Bold colors
    BOLD = "\033[1m"
    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_BLUE = "\033[1;34m"
    BOLD_MAGENTA = "\033[1;35m"
    BOLD_CYAN = "\033[1;36m"
    BOLD_WHITE = "\033[1;37m"

    # Dim
    DIM = "\033[2m"

    @staticmethod
    def is_terminal() -> bool:
        """Check if output is a terminal (supports colors)."""
        return sys.stdout.isatty()

    @staticmethod
    def colorize(text: str, color: str, use_color: bool = None) -> str:
        """
        Colorize text if terminal supports it.

        Args:
            text: Text to colorize
            color: Color code from Colors class
            use_color: Force color on/off, or None to auto-detect

        Returns:
            Colorized text
        """
        if use_color is None:
            use_color = Colors.is_terminal()

        if use_color:
            return f"{color}{text}{Colors.RESET}"
        return text


def format_state_with_colors(
    state_string: str,
    use_color: bool = None
) -> str:
    """
    Add ANSI colors to a proof state string.

    Args:
        state_string: Plain text state string
        use_color: Force color on/off, or None to auto-detect

    Returns:
        Colorized state string
    """
    if use_color is None:
        use_color = Colors.is_terminal()

    if not use_color:
        return state_string

    lines = state_string.split("\n")
    colored_lines = []

    for line in lines:
        # Color status lines
        if line.startswith("Status:"):
            if "Complete" in line or "✓" in line:
                colored_lines.append(Colors.colorize(line, Colors.BOLD_GREEN))
            elif "In progress" in line:
                colored_lines.append(Colors.colorize(line, Colors.BOLD_YELLOW))
            else:
                colored_lines.append(line)

        # Color goal headers
        elif line.startswith("Goal"):
            colored_lines.append(Colors.colorize(line, Colors.BOLD_BLUE))

        # Color section headers
        elif line.startswith("Hypotheses:") or line.startswith("Target:"):
            colored_lines.append(Colors.colorize(line, Colors.BOLD_CYAN))

        # Color errors
        elif line.startswith("Errors:") or "Error" in line:
            colored_lines.append(Colors.colorize(line, Colors.BOLD_RED))

        # Color messages
        elif line.startswith("Messages:"):
            colored_lines.append(Colors.colorize(line, Colors.BOLD_MAGENTA))

        # Color proof finished message
        elif "Proof finished" in line or "proof complete" in line.lower():
            colored_lines.append(Colors.colorize(line, Colors.BOLD_GREEN))

        # Color turnstile symbol
        elif "⊢" in line:
            colored_line = line.replace("⊢", Colors.colorize("⊢", Colors.GREEN))
            colored_lines.append(colored_line)

        else:
            colored_lines.append(line)

    return "\n".join(colored_lines)


def parse_theorem_statement(theorem: str) -> Dict[str, str]:
    """
    Parse a theorem statement into components.

    Args:
        theorem: Theorem string (e.g., "theorem ex : 1 + 1 = 2 := by sorry")

    Returns:
        Dictionary with 'name', 'statement', and 'proof_start' keys

    Example:
        >>> parse_theorem_statement("theorem add_comm : ∀ n m, n + m = m + n := by sorry")
        {'name': 'add_comm', 'statement': '∀ n m, n + m = m + n', 'proof_start': 'sorry'}
    """
    result = {
        "name": "",
        "statement": "",
        "proof_start": ""
    }

    # Match pattern: theorem <name> : <statement> := by <proof>
    pattern = r"theorem\s+(\w+)\s*:\s*(.+?)\s*:=\s*by\s*(.+)"
    match = re.match(pattern, theorem.strip())

    if match:
        result["name"] = match.group(1)
        result["statement"] = match.group(2).strip()
        result["proof_start"] = match.group(3).strip()

    return result


def validate_theorem_syntax(theorem: str) -> tuple[bool, Optional[str]]:
    """
    Validate basic theorem syntax.

    Args:
        theorem: Theorem statement to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if it starts with 'theorem'
    if not theorem.strip().startswith("theorem"):
        return False, "Theorem must start with 'theorem' keyword"

    # Check if it contains ':'
    if ":" not in theorem:
        return False, "Theorem must contain ':' separating name and statement"

    # Check if it contains ':= by'
    if ":= by" not in theorem and ":=by" not in theorem:
        return False, "Theorem must contain ':= by' before proof"

    return True, None


def format_tactic_error(error_message: str) -> str:
    """
    Format a tactic error message for better readability.

    Args:
        error_message: Raw error message from Lean

    Returns:
        Formatted error message
    """
    # Remove excessive whitespace
    formatted = re.sub(r'\s+', ' ', error_message.strip())

    # Add line breaks before common error indicators
    formatted = re.sub(r'(error:)', r'\n\1', formatted)
    formatted = re.sub(r'(expected)', r'\n  \1', formatted)
    formatted = re.sub(r'(found)', r'\n  \1', formatted)

    return formatted


def format_hypothesis(hyp: str) -> str:
    """
    Format a hypothesis for display.

    Args:
        hyp: Raw hypothesis string

    Returns:
        Formatted hypothesis
    """
    # Clean up spacing
    formatted = hyp.strip()

    # Add spacing around ':' if needed
    if ':' in formatted and ' : ' not in formatted:
        formatted = formatted.replace(':', ' : ')

    return formatted


def format_target(target: str) -> str:
    """
    Format a target goal for display.

    Args:
        target: Raw target string

    Returns:
        Formatted target
    """
    # Clean up spacing
    formatted = target.strip()

    # Ensure turnstile is present
    if not formatted.startswith("⊢"):
        formatted = f"⊢ {formatted}"

    return formatted


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def format_tactic_list(tactics: list[str], max_display: int = 10) -> str:
    """
    Format a list of tactics for display.

    Args:
        tactics: List of tactic names
        max_display: Maximum number to display before truncating

    Returns:
        Formatted string
    """
    if not tactics:
        return "No tactics applied yet"

    if len(tactics) <= max_display:
        return ", ".join(tactics)

    displayed = tactics[-max_display:]
    num_hidden = len(tactics) - max_display
    return f"[{num_hidden} more...], {', '.join(displayed)}"


def indent_text(text: str, indent: int = 2) -> str:
    """
    Indent all lines in a text block.

    Args:
        text: Text to indent
        indent: Number of spaces to indent

    Returns:
        Indented text
    """
    prefix = " " * indent
    lines = text.split("\n")
    return "\n".join(prefix + line if line.strip() else line for line in lines)
