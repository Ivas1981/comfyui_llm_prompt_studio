import logging
import traceback

_logger = logging.getLogger("llm_prompt_studio")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[LLMPromptStudio] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

# Register nodes first so a failure in the (optional) HTTP routes can never wipe the
# whole node pack. Each step is isolated so we surface the precise failure.
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    _logger.error("ERROR loading nodes, traceback:")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    WEB_DIRECTORY = "./web"
    _logger.info("Package loaded: Writer, Critic, Smart Save, Library Loader, "
                 "Scene Builder, Smart Loader, Multi-CLIP SDXL.")

try:
    from . import server_routes  # noqa: F401 — registers HTTP routes /llm_prompt_studio/*
except Exception:
    # The routes need ComfyUI's `server` module; absent only outside ComfyUI. Nodes still work.
    _logger.warning("HTTP routes (server_routes) not loaded: %s", traceback.format_exc().strip())