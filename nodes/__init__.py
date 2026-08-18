from .writer import LLMPromptStudioWriter
from .critic import LLMPromptStudioCritic
from .smart_save import LLMPromptStudioSmartSave
from .library_loader import LLMPromptStudioLibraryLoader
from .scene_builder import LLMPromptStudioSceneBuilder
from .smart_loader import LLMPromptStudioSmartLoader
from .multi_clip import LLMPromptStudioMultiClipSDXL
from .smart_parameters import LLMPromptStudioSmartParameters
from .ksampler_hiresfix import LLMPromptStudioKSamplerHiresFix
from .face_detailer import LLMPromptStudioFaceDetailer

NODE_CLASS_MAPPINGS = {
    "LLMPromptStudioWriter": LLMPromptStudioWriter,
    "LLMPromptStudioCritic": LLMPromptStudioCritic,
    "LLMPromptStudioSmartSave": LLMPromptStudioSmartSave,
    "LLMPromptStudioLibraryLoader": LLMPromptStudioLibraryLoader,
    "LLMPromptStudioSceneBuilder": LLMPromptStudioSceneBuilder,
    "LLMPromptStudioSmartLoader": LLMPromptStudioSmartLoader,
    "LLMPromptStudioMultiClipSDXL": LLMPromptStudioMultiClipSDXL,
    "LLMPromptStudioSmartParameters": LLMPromptStudioSmartParameters,
    "LLMPromptStudioKSamplerHiresFix": LLMPromptStudioKSamplerHiresFix,
    "LLMPromptStudioFaceDetailer": LLMPromptStudioFaceDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptStudioWriter": "LLM Prompt Studio Writer",
    "LLMPromptStudioCritic": "LLM Prompt Studio Image Critic",
    "LLMPromptStudioSmartSave": "LLM Prompt Studio Smart Save",
    "LLMPromptStudioLibraryLoader": "LLM Prompt Studio Library Loader",
    "LLMPromptStudioSceneBuilder": "LLM Prompt Studio Scene Builder",
    "LLMPromptStudioSmartLoader": "LLM Prompt Studio Smart Loader",
    "LLMPromptStudioMultiClipSDXL": "LLM Prompt Studio Smart Multi-Clip",
    "LLMPromptStudioSmartParameters": "LLM Prompt Studio Smart Parameters",
    "LLMPromptStudioKSamplerHiresFix": "LLM Prompt Studio KSampler (Hires Fix)",
    "LLMPromptStudioFaceDetailer": "LLM Prompt Studio Face Detailer",
}
