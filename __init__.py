from .nodes.loader import RunningHubVoxCPMLoadModel
from .nodes.generate import RunningHubVoxCPMGenerate

NODE_CLASS_MAPPINGS = {
    "RunningHub_VoxCPM_LoadModel": RunningHubVoxCPMLoadModel,
    "RunningHub_VoxCPM_Generate": RunningHubVoxCPMGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub_VoxCPM_LoadModel": "RunningHub VoxCPM Load Model",
    "RunningHub_VoxCPM_Generate": "RunningHub VoxCPM Generate Speech",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
