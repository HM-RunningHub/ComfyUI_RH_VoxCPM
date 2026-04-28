from .nodes.loader import RunningHubVoxCPMLoadModel
from .nodes.generate import RunningHubVoxCPMGenerate
from .nodes.multi_speaker import (
    RunningHubVoxCPMMultiSpeaker,
    RunningHubVoxCPMMultiSpeakerListReference,
)
from .nodes.train import (
    RunningHubVoxCPMDatasetEntry,
    RunningHubVoxCPMDatasetBuild,
    RunningHubVoxCPMDatasetBuildBatch,
    RunningHubVoxCPMTrainLoRA,
    # RunningHubVoxCPMTrainFull,
)

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "RunningHub_VoxCPM_LoadModel": RunningHubVoxCPMLoadModel,
    "RunningHub_VoxCPM_Generate": RunningHubVoxCPMGenerate,
    "RunningHub_VoxCPM_MultiSpeaker": RunningHubVoxCPMMultiSpeaker,
    "RunningHub_VoxCPM_MultiSpeaker_ListReference": RunningHubVoxCPMMultiSpeakerListReference,
    "RunningHub_VoxCPM_DatasetEntry": RunningHubVoxCPMDatasetEntry,
    "RunningHub_VoxCPM_DatasetBuild": RunningHubVoxCPMDatasetBuild,
    "RunningHub_VoxCPM_DatasetBuildBatch": RunningHubVoxCPMDatasetBuildBatch,
    "RunningHub_VoxCPM_TrainLoRA": RunningHubVoxCPMTrainLoRA,
    # "RunningHub_VoxCPM_TrainFull": RunningHubVoxCPMTrainFull,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub_VoxCPM_LoadModel": "RunningHub VoxCPM Load Model",
    "RunningHub_VoxCPM_Generate": "RunningHub VoxCPM Generate Speech",
    "RunningHub_VoxCPM_MultiSpeaker": "RunningHub VoxCPM Multi-Speaker",
    "RunningHub_VoxCPM_MultiSpeaker_ListReference": "RunningHub VoxCPM Multi-Speaker (Dynamic Audio)",
    "RunningHub_VoxCPM_DatasetEntry": "RunningHub VoxCPM Dataset Entry",
    "RunningHub_VoxCPM_DatasetBuild": "RunningHub VoxCPM Dataset Build",
    "RunningHub_VoxCPM_DatasetBuildBatch": "RunningHub VoxCPM Dataset Build (Batch)",
    "RunningHub_VoxCPM_TrainLoRA": "RunningHub VoxCPM Train LoRA",
    # "RunningHub_VoxCPM_TrainFull": "RunningHub VoxCPM Train Full",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
