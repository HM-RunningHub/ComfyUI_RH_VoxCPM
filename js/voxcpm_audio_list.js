import { app } from "../../scripts/app.js";

const NODE_NAME = "RunningHub_VoxCPM_MultiSpeaker_ListReference";
const INPUT_PREFIX = "audio_";
const INPUT_TYPE = "AUDIO";
const MIN_INPUTS = 2;

function getAudioInputs(node) {
    return (node.inputs || []).filter((input) => input.name?.startsWith(INPUT_PREFIX));
}

function isInputConnected(input) {
    if (!input) {
        return false;
    }
    if (input.link != null) {
        return true;
    }
    if (Array.isArray(input.links) && input.links.length > 0) {
        return true;
    }
    return false;
}

function renameAudioInputs(node) {
    const audioInputs = getAudioInputs(node);
    audioInputs.forEach((input, index) => {
        input.name = `${INPUT_PREFIX}${index + 1}`;
        input.label = input.name;
        input.type = INPUT_TYPE;
    });
}

function ensureMinInputs(node) {
    let audioInputs = getAudioInputs(node);
    while (audioInputs.length < MIN_INPUTS) {
        node.addInput(`${INPUT_PREFIX}${audioInputs.length + 1}`, INPUT_TYPE);
        audioInputs = getAudioInputs(node);
    }
    renameAudioInputs(node);
}

function ensureSpareInput(node) {
    const audioInputs = getAudioInputs(node);
    const hasEmptySlot = audioInputs.some((input) => !isInputConnected(input));
    if (!hasEmptySlot) {
        node.addInput(`${INPUT_PREFIX}${audioInputs.length + 1}`, INPUT_TYPE);
        renameAudioInputs(node);
    }
}

function trimTrailingInputs(node) {
    while (true) {
        const audioInputs = getAudioInputs(node);
        if (audioInputs.length <= MIN_INPUTS) {
            break;
        }

        const lastInput = audioInputs[audioInputs.length - 1];
        if (isInputConnected(lastInput)) {
            break;
        }

        const lastIndex = node.inputs.indexOf(lastInput);
        if (lastIndex < 0) {
            break;
        }
        node.removeInput(lastIndex);
    }
    renameAudioInputs(node);
}

function syncInputs(node) {
    ensureMinInputs(node);
    trimTrailingInputs(node);
    ensureSpareInput(node);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function scheduleSync(node) {
    if (!node) {
        return;
    }
    if (node.__rhVoxCPMSyncTimer) {
        clearTimeout(node.__rhVoxCPMSyncTimer);
    }
    node.__rhVoxCPMSyncTimer = setTimeout(() => {
        node.__rhVoxCPMSyncTimer = null;
        syncInputs(node);
    }, 0);
}

app.registerExtension({
    name: "RunningHub.VoxCPM.MultiSpeakerListReferenceInputs",
    rh: {
        type: "nodes",
        nodes: [NODE_NAME],
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            scheduleSync(this);
            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = originalOnConnectionsChange?.apply(this, arguments);
            scheduleSync(this);
            return result;
        };

        const originalOnConnectInput = nodeType.prototype.onConnectInput;
        nodeType.prototype.onConnectInput = function () {
            const result = originalOnConnectInput?.apply(this, arguments);
            scheduleSync(this);
            return result;
        };
    },

    nodeCreated(node) {
        if (node.comfyClass === NODE_NAME) {
            scheduleSync(node);
        }
    },

    loadedGraphNode(node) {
        if (node.comfyClass === NODE_NAME) {
            scheduleSync(node);
            setTimeout(() => scheduleSync(node), 50);
        }
    },
});
