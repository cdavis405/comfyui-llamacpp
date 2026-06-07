from .llamacpp_nodes import (
    LlamaCppChat,
    LlamaCppLoadImagesFromFolder,
    LlamaCppSaveText,
    LlamaCppUnload,
)

NODE_CLASS_MAPPINGS = {
    "LlamaCppChat": LlamaCppChat,
    "LlamaCppLoadImagesFromFolder": LlamaCppLoadImagesFromFolder,
    "LlamaCppSaveText": LlamaCppSaveText,
    "LlamaCppUnload": LlamaCppUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppChat": "llama.cpp Chat",
    "LlamaCppLoadImagesFromFolder": "llama.cpp Load Images From Folder",
    "LlamaCppSaveText": "llama.cpp Save Text",
    "LlamaCppUnload": "llama.cpp Unload Server",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
