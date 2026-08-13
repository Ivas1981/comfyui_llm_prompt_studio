from ..library import library_scenes, load_library, resolve_library_path
from ..debug import node_span


class LLMPromptStudioLibraryLoader:
    """Loads a saved scene from the prompt library."""
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library_path": ("STRING", {"default": "llm_prompt_studio_library.json"}),
                "scene_name": (library_scenes(),),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "scene_name",
                    "face_positive", "face_negative")

    def load(self, library_path, scene_name, unique_id=None):
        with node_span("Library Loader", unique_id, {"scene_name": scene_name}):
            entries = load_library(resolve_library_path(library_path))
            if not entries:
                raise RuntimeError(
                    "The library is empty — save at least one prompt via "
                    "LLM Prompt Studio Smart Save first.")
            for e in entries:
                if isinstance(e, dict) and e.get("name") == scene_name:
                    return (str(e.get("prompt", "")),
                            str(e.get("negative_prompt", "")),
                            str(e.get("name", "")),
                            str(e.get("face_positive", "")),
                            str(e.get("face_negative", "")))
            raise RuntimeError(f"Scene '{scene_name}' not found in the library.")