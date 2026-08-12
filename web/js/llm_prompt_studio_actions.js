import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import {
    getW, getJSON, postJSON,
    isWriter, isCritic, isScene,
    PH_DOWN, PH_EMPTY, LIB_EMPTY, lastSaveData,
} from "./llm_prompt_studio_shared.js";

// ---------------------------------------------------------------------------
// Button handlers
// ---------------------------------------------------------------------------

// Apply the refreshed model list into a cached node definition. ComfyUI stores the
// combo values in different shapes across versions, so we try each known one and
// update whichever holds the list. Returns true if something was patched.
function patchDefModelList(def, models) {
    if (!def) return false;
    const input = def.input || def.inputs;
    if (!input) return false;
    const required = input.required || input.input_required || {};
    const optional = input.optional || {};
    const slot = required.model || optional.model;
    if (!slot) return false;

    let patched = false;
    // Shape A (older ComfyUI): slot === [valuesArray, ...]
    if (Array.isArray(slot[0])) {
        slot[0] = models;
        patched = true;
    }
    // Shape B (newer ComfyUI): slot === ["COMBO", { options: { values: [...] } }]
    if (Array.isArray(slot) && slot.length > 1 && slot[1] && typeof slot[1] === "object") {
        const cfg = slot[1];
        if (cfg.options && Array.isArray(cfg.options.values)) {
            cfg.options.values = models;
            patched = true;
        }
        if (Array.isArray(cfg.values)) {
            cfg.values = models;
            patched = true;
        }
    }
    // Fallback: a plain combo object exposing .values
    if (slot.values && Array.isArray(slot.values)) {
        slot.values = models;
        patched = true;
    }
    return patched;
}

export async function refreshModels(node) {
    const server_url = getW(node, "server_url")?.value || "http://localhost:1234/v1";
    const api_key = getW(node, "api_key")?.value || "";
    let models = [];
    let error = null;
    try {
        const data = await postJSON("/llm_prompt_studio/models", { server_url, api_key });
        models = data.models || [];
        error = data.error || null;
    } catch (e) {
        error = String(e);
    }
    if (!models.length) models = error ? [PH_DOWN] : [PH_EMPTY];

    // 1) Patch every live node's model widget.
    for (const n of app.graph.nodes) {
        const mw = getW(n, "model");
        if (mw && (isWriter(n) || isCritic(n) || isScene(n))) {
            mw.options = mw.options || {};
            mw.options.values = models;
            if (!models.includes(mw.value)) mw.value = models[0];
        }
    }

    // 2) Update the cached node definitions (app.nodeDefs, fed by /object_info at page
    //    load). Without this, loading a SAVED workflow would still validate the model
    //    value against the stale placeholder list and throw "Value not in list".
    //    ComfyUI's nodeDefs shape has changed across versions, so try several known
    //    layouts and bail out gracefully (never throw) if none match.
    const modelNodes = ["LLMPromptStudioWriter", "LLMPromptStudioCritic", "LLMPromptStudioSceneBuilder"];
    for (const t of modelNodes) {
        try {
            patchDefModelList(app && app.nodeDefs && app.nodeDefs[t], models);
        } catch (e) {
            console.warn("[LLMPromptStudio] Could not refresh cached model list for", t, e);
        }
    }

    app.graph.setDirtyCanvas(true, true);
}

export async function saveToLibrary(node) {
    const data = lastSaveData.get(node.id) || {};
    const library_path = getW(node, "library_path")?.value || "";
    if (!data.positive) {
        alert("Nothing to save yet — run the node with an approved image first.");
        return;
    }
    try {
        const res = await postJSON("/llm_prompt_studio/library/save", {
            library_path,
            scene_name: data.scene_name || "",
            positive: data.positive || "",
            negative: data.negative || "",
            face_positive: data.face_positive || "",
            face_negative: data.face_negative || "",
        });
        if (res.error) {
            alert("Library save failed: " + res.error);
        } else {
            alert(res.added ? ("Saved to library as: " + res.name)
                            : ("Already in library: " + res.name));
        }
    } catch (e) {
        alert("Library save failed: " + e);
    }
}

export async function refreshScenes(node) {
    const library_path = getW(node, "library_path")?.value || "";
    let scenes = [];
    try {
        const data = await getJSON(
            "/llm_prompt_studio/library/scenes?library_path=" + encodeURIComponent(library_path));
        scenes = data.scenes || [];
    } catch (e) {
        scenes = [];
    }
    if (!scenes.length) scenes = [LIB_EMPTY];

    const sw = getW(node, "scene_name");
    if (sw) {
        sw.options = sw.options || {};
        sw.options.values = scenes;
        if (!scenes.includes(sw.value)) sw.value = scenes[0];
    }
    app.graph.setDirtyCanvas(true, true);
}

export function sendToWriter(node) {
    const text = (getW(node, "description_view")?.value ||
                  getW(node, "prompt_view")?.value || "");
    if (!text.trim()) {
        alert("Nothing to send — run stage 1 (Describe) first.");
        return;
    }
    const writer = app.graph.nodes.find(isWriter);
    if (!writer) {
        alert("No LLM Prompt Studio Writer node found in the graph.");
        return;
    }
    const ideaW = getW(writer, "idea");
    if (ideaW) ideaW.value = text;
    app.graph.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Re-queue helper for the auto-revision loop
// ---------------------------------------------------------------------------
export function requeuePrompt() {
    try {
        if (typeof app.queuePrompt === "function") {
            app.queuePrompt();
        } else if (typeof api.queuePrompt === "function") {
            api.queuePrompt();
        } else {
            console.warn("[LLMPromptStudio.Bridge] No queuePrompt method available.");
        }
    } catch (err) {
        console.error("[LLMPromptStudio.Bridge] Failed to re-queue prompt:", err);
    }
}