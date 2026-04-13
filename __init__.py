from .nodes.loader import RunningHubVoxCPMLoadModel
from .nodes.generate import RunningHubVoxCPMGenerate
from .nodes.multi_speaker import (
    RunningHubVoxCPMMultiSpeaker,
    RunningHubVoxCPMMultiSpeakerListReference,
)

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "RunningHub_VoxCPM_LoadModel": RunningHubVoxCPMLoadModel,
    "RunningHub_VoxCPM_Generate": RunningHubVoxCPMGenerate,
    "RunningHub_VoxCPM_MultiSpeaker": RunningHubVoxCPMMultiSpeaker,
    "RunningHub_VoxCPM_MultiSpeaker_ListReference": RunningHubVoxCPMMultiSpeakerListReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub_VoxCPM_LoadModel": "RunningHub VoxCPM Load Model",
    "RunningHub_VoxCPM_Generate": "RunningHub VoxCPM Generate Speech",
    "RunningHub_VoxCPM_MultiSpeaker": "RunningHub VoxCPM Multi-Speaker",
    "RunningHub_VoxCPM_MultiSpeaker_ListReference": "RunningHub VoxCPM Multi-Speaker (Dynamic Audio)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
