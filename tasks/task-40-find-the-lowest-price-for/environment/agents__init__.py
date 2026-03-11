"""Agent implementations for Harbor."""
from .base import BaseAgent
from .cocoa_agent import CocoaAgent

try:
    from .skill_agent import SkillAgent
except ImportError:
    SkillAgent = None

# CLI/API agents — safe imports (no special deps beyond what's installed)
try:
    from .claude_code_agent import ClaudeCodeAgent
except ImportError:
    ClaudeCodeAgent = None

try:
    from .codex_agent import CodexAgent
except ImportError:
    CodexAgent = None

try:
    from .gemini_cli_agent import GeminiCLIAgent
except ImportError:
    GeminiCLIAgent = None

try:
    from .manus_agent import ManusAgent
except ImportError:
    ManusAgent = None

try:
    from .openai_deep_research_agent import OpenAIDeepResearchAgent
except ImportError:
    OpenAIDeepResearchAgent = None

try:
    from .gemini_deep_research_agent import GeminiDeepResearchAgent
except ImportError:
    GeminiDeepResearchAgent = None

__all__ = ["BaseAgent", "CocoaAgent"]
