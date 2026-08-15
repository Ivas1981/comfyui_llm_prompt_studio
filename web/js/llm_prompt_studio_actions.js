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

// The four model-load widgets that the "⚙ Advanced settings" button shows/hides.
const ADVANCED_WIDGETS = ["context_length", "gpu_offload",
                           "flash_attention", "offload_kv_cache_to_gpu",
                           "temperature", "max_tokens", "repeat_penalty",
                           "top_k", "top_p", "min_p"];

// Hide/show a single widget named `name` on `node`.
//
// ComfyUI has two very different front-ends and we must support both:
//  - Modern (Nodes 2.0 / Vue): widgets are rendered by Vue components, so `widget.element`
//    is `undefined` and `element.style.display` is ignored. The renderer reads the canonical
//    `hidden` / `options.hidden` flags, so we set those to collapse the widget + its layout.
//  - Legacy (canvas / DOM widgets, e.g. 0.19.x): `widget.element` exists, so we also flip
//    `element.style.display` as belt-and-braces. The VALUE is always preserved (hiding never
//    clears it), so the node still receives context_length / gpu_offload / etc.
//
// `computeSize` is overridden to [0, -4] while hidden (collapsing the layout gap) and
// restored from the cache when shown again.
const _advOrig = {};   // "<nodeId>:<name>" -> original computeSize

function setWidgetHidden(node, name, hidden) {
    const w = getW(node, name);
    if (!w) return;
    const key = node.id + ":" + name;
    if (hidden) {
        if (!_advOrig[key] && typeof w.computeSize === "function") {
            _advOrig[key] = w.computeSize.bind(w);
        }
        w.hidden = true;
        if (w.options) w.options.hidden = true;
        if (typeof w.computeSize === "function") w.computeSize = () => [0, -4];
    } else {
        const orig = _advOrig[key];
        w.hidden = false;
        if (w.options) w.options.hidden = false;
        if (orig) w.computeSize = orig;
    }
    // Belt-and-braces for legacy front-ends that own the DOM element directly.
    for (const k of ["element", "inputEl"]) {
        const el = w[k];
        if (el && el.style) el.style.display = hidden ? "none" : "";
    }
}

export function setAdvancedCollapsed(node, collapsed) {
    node._advancedCollapsed = collapsed;
    for (const name of ADVANCED_WIDGETS) {
        setWidgetHidden(node, name, collapsed);
    }
    const btn = node.widgets && node.widgets.find(w => w.name === "⚙ Advanced settings");
    if (btn) btn.label = collapsed ? "⚙ Advanced settings ▸" : "⚙ Advanced settings ▾";
    // Refit the node so a hidden/shown block doesn't leave a stale gap. Keep the current
    // width so the node is never squished horizontally; only the height follows the layout.
    try {
        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            const sz = node.computeSize();
            node.setSize([node.size[0], sz[1]]);
        }
        if (app.graph) app.graph.setDirtyCanvas(true, true);
    } catch (e) {
        console.warn("[LLMPromptStudio] Advanced settings reflow failed:", e);
    }
}

export function toggleAdvancedSettings(node) {
    setAdvancedCollapsed(node, !node._advancedCollapsed);
}

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

    // 1) Patch every live node's model widget — but ONLY for nodes whose server_url
    //    matches the server we just refreshed. Refreshing server B must not clobber
    //    server A's model combo (the cross-server mix-up that made the wrong model load).
    if (app.graph && app.graph.nodes) {
        for (const n of app.graph.nodes) {
            if (!(isWriter(n) || isCritic(n) || isScene(n))) continue;
            const nServer = getW(n, "server_url")?.value || "http://localhost:1234/v1";
            if (nServer !== server_url) continue;
            const mw = getW(n, "model");
            if (mw) {
                mw.options = mw.options || {};
                mw.options.values = models;
                if (!models.includes(mw.value)) mw.value = models[0];
            }
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
// Preset management (Writer style presets)
// ---------------------------------------------------------------------------
export async function reloadPresets(node) {
    try {
        const data = await getJSON("/llm_prompt_studio/presets");
        const names = data.names || [];
        const w = getW(node, "style_preset");
        if (!w) return;
        const opts = ["— none —", ...names];
        w.options = w.options || {};
        w.options.values = opts;
        if (!opts.includes(w.value)) w.value = "— none —";
        app.graph.setDirtyCanvas(true, true);
    } catch (e) {
        alert("Could not reload presets: " + e);
    }
}

export async function resetPresets(node) {
    if (typeof confirm === "function" &&
        !confirm("Reset style presets to defaults? Your custom edits will be lost.")) {
        return;
    }
    try {
        const data = await postJSON("/llm_prompt_studio/presets/reset", {});
        if (data.error) {
            alert("Preset reset failed: " + data.error);
            return;
        }
        await reloadPresets(node);
        alert("Presets reset to defaults.");
    } catch (e) {
        alert("Preset reset failed: " + e);
    }
}

export async function copyPresetsPath(node) {
    try {
        const data = await getJSON("/llm_prompt_studio/presets");
        const path = data.path || "";
        if (typeof navigator !== "undefined" && navigator.clipboard) {
            navigator.clipboard.writeText(path).catch(() => {});
        }
        alert("Presets file path copied:\n" + path);
    } catch (e) {
        alert("Could not get presets path: " + e);
    }
}

export function requeuePrompt() {
    try {
        if (typeof app.queuePrompt === "function") {
            try {
                // Modern front-ends expect the serialized graph + options; mirror the built-in
                // "Queue Prompt" button so auto-revision re-queues correctly.
                app.queuePrompt(app.graph.serialize(), {});
            } catch (e) {
                // Fallback for older front-ends (e.g. 0.19.3) whose queuePrompt takes no args.
                app.queuePrompt();
            }
        } else if (typeof api.queuePrompt === "function") {
            api.queuePrompt();
        } else {
            console.warn("[LLMPromptStudio.Bridge] No queuePrompt method available.");
        }
    } catch (err) {
        console.error("[LLMPromptStudio.Bridge] Failed to re-queue prompt:", err);
    }
}