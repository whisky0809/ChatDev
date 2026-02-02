"""Register built-in agent providers."""

from runtime.node.agent.providers.base import ProviderRegistry

from runtime.node.agent.providers.openai_provider import OpenAIProvider

ProviderRegistry.register(
    "openai",
    OpenAIProvider,
    label="OpenAI",
    summary="OpenAI models via the official OpenAI SDK (responses API)",
)

try:
    from runtime.node.agent.providers.moonshot_provider import MoonshotProvider
except ImportError:
    MoonshotProvider = None

if MoonshotProvider is not None:
    ProviderRegistry.register(
        "moonshot",
        MoonshotProvider,
        label="Moonshot AI",
        summary="Moonshot AI (Kimi) models via the OpenAI-compatible chat completions API",
    )

# Gemini provider (optional dependency)
try:
    from runtime.node.agent.providers.gemini_provider import GeminiProvider
except ImportError:
    GeminiProvider = None

if GeminiProvider is not None:
    ProviderRegistry.register(
        "gemini",
        GeminiProvider,
        label="Google Gemini",
        summary="Google Gemini models via google-genai",
    )
else:
    print("Gemini provider not registered: google-genai library not found.")
