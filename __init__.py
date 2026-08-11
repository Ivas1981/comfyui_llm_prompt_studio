import logging
import traceback

_logger = logging.getLogger("llm_prompt_studio")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[LLMPromptStudio] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

try:
    from . import server_routes  # noqa: F401 — registers HTTP routes /llm_prompt_studio/*
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    WEB_DIRECTORY = "./web"
    _logger.info("Package loaded: Writer, Critic, Smart Save, Library Loader, "
                 "Scene Builder, Smart Loader, Multi-CLIP SDXL.")
except Exception:
    _logger.error("ERROR loading package, traceback:")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}