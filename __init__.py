from .nodes.loader import VoxCPMLoadModel
from .nodes.generate import VoxCPMGenerate

NODE_CLASS_MAPPINGS = {
    "VoxCPM_LoadModel": VoxCPMLoadModel,
    "VoxCPM_Generate": VoxCPMGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxCPM_LoadModel": "VoxCPM: Load Model",
    "VoxCPM_Generate": "VoxCPM: Generate Speech",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
