import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { getJSON } from "./llm_prompt_studio_shared.js";

// Dynamic class combos for Face Detailer: when the user picks a YOLO seg / gender
// model, the matching class widget (yolo_seg_class / gender_model_female_class)
// is repopulated with that model's real class names (from the /model_classes API).
// ComfyUI core INPUT_TYPES cannot make one widget's options depend on another at
// runtime, so this reactivity lives in the front-end.
const NODE_TYPE = "LLMPromptStudioFaceDetailer";

function getW(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

async function refreshClassOptions(modelWidget, classWidget, kind) {
    if (!modelWidget || !classWidget) return;
    const modelName = modelWidget.value;
    if (!modelName || modelName === "(none)") {
        // No model selected: keep safe defaults so the combo stays valid.
        const fallback = kind === "gender" ? ["female", "male"] : ["face"];
        classWidget.options.values = fallback;
        if (!fallback.includes(classWidget.value)) classWidget.value = fallback[0];
        return;
    }
    try {
        const data = await getJSON(
            `/llm_prompt_studio/model_classes?model=${encodeURIComponent(modelName)}&kind=${kind}`
        );
        const names = (data && data.names) || [];
        if (!names.length) return; // server couldn't list -> keep current options
        const opts = names.map((n) => n.name);
        classWidget.options.values = opts;
        if (!opts.includes(classWidget.value)) {
            classWidget.value = opts[0];
        }
    } catch (e) {
        // keep current options on any error
    }
}

app.registerExtension({
    name: "llm_prompt_studio.classes",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            const segModel = getW(this, "yolo_seg_model_name");
            const segClass = getW(this, "yolo_seg_class");
            const genderModel = getW(this, "gender_model_name");
            const genderClass = getW(this, "gender_model_female_class");

            if (segModel && segClass) {
                refreshClassOptions(segModel, segClass, "seg");
                const orig = segModel.callback;
                segModel.callback = function () {
                    refreshClassOptions(segModel, segClass, "seg");
                    if (orig) return orig.apply(this, arguments);
                };
            }
            if (genderModel && genderClass) {
                refreshClassOptions(genderModel, genderClass, "gender");
                const orig = genderModel.callback;
                genderModel.callback = function () {
                    refreshClassOptions(genderModel, genderClass, "gender");
                    if (orig) return orig.apply(this, arguments);
                };
            }
            return r;
        };

        // After a saved workflow is loaded, re-sync the class combos to the now-selected
        // model (the fetch is async, so it runs after ComfyUI has applied widgets_values).
        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            const segModel = getW(this, "yolo_seg_model_name");
            const segClass = getW(this, "yolo_seg_class");
            const genderModel = getW(this, "gender_model_name");
            const genderClass = getW(this, "gender_model_female_class");
            if (segModel && segClass) refreshClassOptions(segModel, segClass, "seg");
            if (genderModel && genderClass) refreshClassOptions(genderModel, genderClass, "gender");
            return r;
        };
    },
});
