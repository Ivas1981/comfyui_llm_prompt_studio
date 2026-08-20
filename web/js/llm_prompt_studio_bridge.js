import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import {
    getW,
    isWriter, isCritic, isSmartSave, isLoader, isScene, isSmartLoader,
    isSmartParams, isKSampler, getJSON,
    lastSaveData, loopCounters,
} from "./llm_prompt_studio_shared.js";

import {
    refreshModels, saveToLibrary, refreshScenes, sendToWriter, requeuePrompt,
    reloadPresets, resetPresets, copyPresetsPath,
    toggleAdvancedSettings, setAdvancedCollapsed, pollServerStatus, setWidgetHidden,
} from "./llm_prompt_studio_actions.js";

// Resolve the Writer that actually produces a Critic's `prompt` input: follow the inbound
// link on the Critic's `prompt` widget back to its origin node. This makes multi-Writer /
// multi-Critic workflows deterministic (each Critic revises its own Writer) instead of
// always looping the first Writer in the graph. Falls back to the first Writer if the link
// can't be resolved.
function upstreamWriter(criticNode) {
    if (!criticNode || !criticNode.inputs || !app.graph || !app.graph.links) return null;
    const promptInput = criticNode.inputs.find((i) => i.name === "prompt");
    if (!promptInput || promptInput.link == null) return null;
    const link = app.graph.links[promptInput.link];
    if (!link) return null;
    const src = app.graph.getNodeById(link[0]);
    if (src && isWriter(src)) return src;
    return null;
}

// ---------------------------------------------------------------------------
// Smart Parameters: recommend sampler params and autofill (unless user edited)
// ---------------------------------------------------------------------------
const SP_EDITABLE = ["steps", "cfg", "sampler_name", "scheduler"];
const SP_TRIGGERS = ["family_override", "preset", "detected_family", "ckpt_name"];

async function autoFillParams(node) {
    const presetW = getW(node, "preset");
    // The "user" preset means "use whatever the user typed" - never overwrite it.
    const preset = presetW ? presetW.value : "user";
    if (preset === "user") return;

    const famW = getW(node, "family_override");
    const detectedW = getW(node, "detected_family");
    const ckptW = getW(node, "ckpt_name");
    // detected_family (when wired from Smart Loader) drives the effective family.
    const family = (detectedW && detectedW.value) ? detectedW.value
                 : (famW ? famW.value : "auto");
    const params = new URLSearchParams({
        family: family || "auto",
        preset: preset,
        ckpt: ckptW ? ckptW.value : "",
    });
    let rec;
    try {
        rec = await getJSON("/llm_prompt_studio/sampler_params?" + params.toString());
    } catch (e) {
        return;  // server unreachable - leave widgets as-is
    }
    if (!rec || rec.error) return;
    node._sp_applying = true;
    try {
        const setIfClean = (name, val) => {
            const w = getW(node, name);
            if (!w || w._sp_dirty) return;
            w.value = val;
        };
        setIfClean("steps", rec.steps);
        setIfClean("cfg", rec.cfg);
        setIfClean("sampler_name", rec.sampler);
        setIfClean("scheduler", rec.scheduler);
    } finally {
        node._sp_applying = false;
    }
    app.graph.setDirtyCanvas(true, true);
}

function setupSmartParams(node) {
    if (node._sp_setup) return;
    node._sp_setup = true;
    // Mark a widget dirty only on genuine user edits (not our programmatic fills).
    for (const name of SP_EDITABLE) {
        const w = getW(node, name);
        if (!w) continue;
        w._sp_dirty = false;
        const orig = w.callback;
        w.callback = function () {
            if (!node._sp_applying) w._sp_dirty = true;
            if (orig) return orig.apply(this, arguments);
        };
    }
    // Recompute recommendations when an input that affects them changes.
    for (const name of SP_TRIGGERS) {
        const w = getW(node, name);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function () {
            const r = orig ? orig.apply(this, arguments) : undefined;
            setTimeout(() => autoFillParams(node), 0);
            return r;
        };
    }
    // When detected_family is connected/disconnected, the effective family changes.
    const origConn = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, connected, link_info) {
        if (origConn) origConn.apply(this, arguments);
        setTimeout(() => autoFillParams(node), 0);
    };
    // Initial fill once the graph is built.
    setTimeout(() => autoFillParams(node), 400);
}

// ---------------------------------------------------------------------------
// KSampler (Hires Fix): hide hires widgets when hires_enabled is off, and hide
// the upscale-model/vis method widgets that don't apply to the chosen
// hires_upscale_type ("latent" vs "latent (model)" vs "pixel (model)").
// ---------------------------------------------------------------------------
const HIRES_FIELDS = [
    "hires_upscale_type", "hires_upscale_method", "hires_latent_upscale_model",
    "hires_latent_upscale_factor", "hires_latent_upscale_tile",
    "hires_upscale_iterations", "hires_steps",
    "hires_cfg", "hires_denoise", "hires_sampler_name", "hires_scheduler",
    "hires_use_same_seed", "hires_seed",
];

function refreshHiresVisibility(node) {
    const enabledW = getW(node, "hires_enabled");
    const enabled = enabledW ? enabledW.value : true;
    const showAll = !enabledW || enabled;
    for (const name of HIRES_FIELDS) {
        setWidgetHidden(node, name, !showAll);
    }
    if (!showAll) return;
    const typeW = getW(node, "hires_upscale_type");
    const t = typeW ? typeW.value : "latent";
    // Only show the widget relevant to the selected upscale type.
    setWidgetHidden(node, "hires_upscale_method", t !== "latent");
    setWidgetHidden(node, "hires_latent_upscale_model", t !== "latent (model)");
    setWidgetHidden(node, "hires_latent_upscale_factor", t === "pixel (model)");
    // The tile size only guards the latent-upscale-model net's memory use.
    setWidgetHidden(node, "hires_latent_upscale_tile", t !== "latent (model)");
}

function setupHires(node) {
    if (node._hires_setup) return;
    node._hires_setup = true;
    const refresh = () => refreshHiresVisibility(node);
    for (const name of ["hires_enabled", "hires_upscale_type"]) {
        const w = getW(node, name);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function () {
            const r = orig ? orig.apply(this, arguments) : undefined;
            setTimeout(refresh, 0);
            return r;
        };
    }
    setTimeout(refresh, 0);
}

// ---------------------------------------------------------------------------
// Button registration
// ---------------------------------------------------------------------------
function addButton(node, label, handler) {
    if (!node.widgets) return;
    if (node.widgets.some(w => w.name === label)) return;  // avoid duplicates
    node.addWidget("button", label, "", handler);
    // The node's canvas size is recomputed so the freshly added button is visible
    // (addWidget grows the layout, but force a redraw to be safe across front-ends).
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
}

// Wire every node's buttons + setup logic. See the timing-independent
// installation block below (processExistingNodes + loadGraphData wrap + the
// `nodeCreated` extension hook).
function setupNodeUI(node) {
    if (!node) return;
    try {
        if (isWriter(node) || isCritic(node)) {
            addButton(node, "Refresh models", () => refreshModels(node));
        }
        if (isWriter(node)) {
            addButton(node, "Reload presets", () => reloadPresets(node));
            addButton(node, "Reset presets", () => resetPresets(node));
            addButton(node, "Copy presets path", () => copyPresetsPath(node));
        }
        if (isScene(node)) {
            addButton(node, "-> Send to Writer", () => sendToWriter(node));
            // Scene Builder also has a model combo, so give it the same manual refresh.
            addButton(node, "Refresh models", () => refreshModels(node));
        }
        if (isWriter(node) || isCritic(node) || isScene(node)) {
            addButton(node, "⚙ Advanced settings", () => toggleAdvancedSettings(node));
            // Collapse by default, but defer so the widget DOM rows exist before we hide them.
            setTimeout(() => setAdvancedCollapsed(node, true), 0);
        }
        if (isSmartSave(node)) {
            addButton(node, "Save prompt to library", () => saveToLibrary(node));
        }
        if (isLoader(node)) {
            addButton(node, "Refresh scene list", () => refreshScenes(node));
        }
        if (isSmartParams(node)) {
            setupSmartParams(node);
        }
        if (isKSampler(node)) {
            setupHires(node);
        }
        // Auto-populate the model list on creation so a saved workflow whose model is not
        // yet in the combo validates without requiring a manual Refresh first. Deferred so it
        // runs after the graph is fully built.
        if (isWriter(node) || isCritic(node) || isScene(node)) {
            setTimeout(() => refreshModels(node), 400);
            setTimeout(() => pollServerStatus(node), 400);
        }
    } catch (e) {
        console.error("[LLMPromptStudio.Bridge] setupNodeUI failed:", e);
    }
}

// ---------------------------------------------------------------------------
// Timing-independent UI installation
// ---------------------------------------------------------------------------
// The node constructor dispatches creation via the `nodeCreated` extension hook
// (it does NOT call a prototype onNodeCreated), so `nodeCreated` is the correct
// hook for nodes created *after* this extension registers. Nodes created *before*
// registration (e.g. the auto-loaded default workflow, which can finish loading
// before this async JS extension is ready) never fire `nodeCreated`. To cover
// those, we also process any already-present nodes and re-process after every
// graph load.
function processExistingNodes() {
    try {
        if (app.graph && app.graph.nodes) {
            for (const n of app.graph.nodes) setupNodeUI(n);
        }
    } catch (e) {
        console.error("[LLMPromptStudio.Bridge] processExistingNodes failed:", e);
    }
}

// Re-apply our UI after any graph is loaded (default workflow, template, paste, etc.).
if (typeof app.loadGraphData === "function") {
    const origLoad = app.loadGraphData.bind(app);
    app.loadGraphData = async function (...args) {
        const r = await origLoad.apply(app, args);
        processExistingNodes();
        return r;
    };
}

// Node types we augment with buttons + per-node UI.
const OUR_NODE_TYPES = [
    "LLMPromptStudioWriter",
    "LLMPromptStudioCritic",
    "LLMPromptStudioSceneBuilder",
    "LLMPromptStudioSmartSave",
    "LLMPromptStudioLibraryLoader",
    "LLMPromptStudioSmartLoader",
    "LLMPromptStudioSmartParameters",
    "LLMPromptStudioKSamplerHiresFix",
];

app.registerExtension({
    name: "llm_prompt_studio.bridge",

    // Buttons + per-node setup for nodes created after this extension registers.
    // Kept as a secondary safety net; the authoritative install happens in
    // beforeRegisterNodeDef (node type onNodeCreated), which also covers the
    // auto-loaded default workflow.
    nodeCreated(node) {
        setupNodeUI(node);
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!OUR_NODE_TYPES.includes(nodeData.name)) return;

        // Install buttons/UI at instance-creation time by wrapping the node type's own
        // onNodeCreated. ComfyUI calls `node.onNodeCreated?.()` inside LGraph.createNode,
        // so this runs for EVERY instance — including nodes from the auto-loaded default
        // workflow, which finish loading before async extension hooks (nodeCreated) have
        // registered. This wrap is installed at node-type registration (startup), before
        // any graph is loaded, so it is immune to the extension-load race. The widgets are
        // present when the Vue node component first renders, so the buttons actually show.
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            setupNodeUI(this);
            return r;
        };

        // Make loading a saved workflow robust: ComfyUI validates each widget's saved value
        // against the combo options at load time, but the model list may not be populated yet
        // (it is fetched asynchronously). Inject the saved value into the combo's allowed
        // options *before* ComfyUI validates, so a saved workflow never trips "Value not in list"
        // even when the server list hasn't arrived. Installed here because this hook runs at
        // node-type registration, before any instance is created.
        if (!["LLMPromptStudioWriter", "LLMPromptStudioCritic",
              "LLMPromptStudioSceneBuilder",
              "LLMPromptStudioSmartParameters"].includes(nodeData.name)) {
            return;
        }
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            const vals = data && data.widgets_values;
            if (this.widgets && Array.isArray(vals)) {
                let vi = 0;
                for (const w of this.widgets) {
                    if (w.type === "button") continue;  // buttons aren't in widgets_values
                    const v = vals[vi];
                    vi++;
                    if (w.name === "model" && w.options && Array.isArray(w.options.values)
                            && v != null && !w.options.values.includes(v)) {
                        w.options.values = w.options.values.concat(v);
                    }
                    // Smart Parameters: a saved workflow may store the "auto" sentinel in
                    // the sampler_name/scheduler combos; inject it so validation passes
                    // before the server list (full scheduler list) is populated.
                    if ((w.name === "sampler_name" || w.name === "scheduler")
                            && w.options && Array.isArray(w.options.values)
                            && v != null && !w.options.values.includes(v)
                            && typeof v === "string" && v !== "") {
                        w.options.values = w.options.values.concat(v);
                    }
                    // release_vram_after_run: a saved workflow can carry a stray non-boolean
                    // (the 5 trailing "" in this node's widgets_values). ComfyUI would assign
                    // that stray value and silently disable the feature, so force the default
                    // (the widget's current value, i.e. True) when the saved value isn't a
                    // real boolean. The Python side also coerces "" -> default as a backstop.
                    if (w.name === "release_vram_after_run"
                            && v !== true && v !== false && w.value !== undefined) {
                        vals[vi] = w.value;
                    }
                }
            }
            return configure.apply(this, arguments);
        };
    },
});

// Process nodes that already exist when this module evaluates (covers a
// late-loading extension where the default workflow finished loading first),
// plus a short safety-net pass after startup in case the graph was still being
// built while this module ran.
processExistingNodes();
setTimeout(processExistingNodes, 800);

// ---------------------------------------------------------------------------
// Executed event: titles, views, stored data and the auto-revision loop
// ---------------------------------------------------------------------------
api.addEventListener("executed", (e) => {
    const d = e.detail || {};
    let node = d.node;
    if (node == null) return;
    if (typeof node === "number" || typeof node === "string") {
        node = app.graph.getNodeById(node);
    }
    if (!node) return;
    const output = d.output || {};

    // --- Image Critic: title, revision view, and the auto-revision loop ---
    if (isCritic(node)) {
        const score = output.score ? output.score[0] : null;
        const approved = output.approved ? output.approved[0] : false;
        const notes = output.revision_notes ? output.revision_notes[0] : "";
        if (score !== null && score !== undefined) {
            node.title = "LLM Prompt Studio Image Critic - score " + score +
                         (approved ? "  approved" : "  rejected");
        }
        const rv = getW(node, "revision_view");
        if (rv) rv.value = notes;
        app.graph.setDirtyCanvas(true, true);

        const auto_loop = output.auto_loop ? output.auto_loop[0] : false;
        if (auto_loop) {
            const max_retries = output.max_retries ? output.max_retries[0] : 3;
            const clear_on_approve = output.clear_notes_on_approve
                ? output.clear_notes_on_approve[0] : true;
            const key = node.id;
            const count = loopCounters.get(key) || 0;

            if (approved) {
                loopCounters.set(key, 0);
                if (clear_on_approve) {
                    for (const n of app.graph.nodes) {
                        if (isWriter(n)) {
                            const rn = getW(n, "revision_notes");
                            if (rn) rn.value = "";
                        }
                    }
                }
            } else if (count < max_retries) {
                const writer = upstreamWriter(node) || app.graph.nodes.find(isWriter);
                if (!writer) {
                    loopCounters.set(key, 0);
                    console.warn("[LLMPromptStudio.Bridge] Auto-loop: no Writer node found, stopping.");
                } else {
                    loopCounters.set(key, count + 1);
                    const rn = getW(writer, "revision_notes");
                    if (rn) rn.value = notes;
                    setTimeout(() => { requeuePrompt(); }, 300);
                }
            } else {
                loopCounters.set(key, 0);
                console.warn("[LLMPromptStudio.Bridge] Auto-loop: max retries (" +
                             max_retries + ") reached, stopping.");
            }
        }
    }

    // --- Smart Save: remember the last saved prompt for the library button ---
    if (isSmartSave(node)) {
        if (output.saved && output.saved[0]) {
            lastSaveData.set(node.id, {
                positive: output.last_positive ? output.last_positive[0] : "",
                negative: output.last_negative ? output.last_negative[0] : "",
                scene_name: output.last_scene_name ? output.last_scene_name[0] : "",
                face_positive: output.last_face_positive ? output.last_face_positive[0] : "",
                face_negative: output.last_face_negative ? output.last_face_negative[0] : "",
            });
            // keep the map bounded
            if (lastSaveData.size > 100) {
                const oldest = lastSaveData.keys().next().value;
                lastSaveData.delete(oldest);
            }
        }
    }

    // --- Smart Loader: detected family in the title + provenance widget ---
    if (isSmartLoader(node)) {
        const family = output.family ? output.family[0] : "";
        const info = output.family_info ? output.family_info[0] : "";
        if (family) {
            node.title = "LLM Prompt Studio Smart Loader - family: " + family;
            app.graph.setDirtyCanvas(true, true);
        }
        const fiw = getW(node, "detected_family_info");
        if (fiw && info) fiw.value = info;
    }

    // --- Scene Builder: show stage results in the view widgets ---
    if (isScene(node)) {
        if (output.description) {
            const dv = getW(node, "description_view");
            if (dv) dv.value = output.description[0];
            app.graph.setDirtyCanvas(true, true);
        }
        if (output.prompt_view) {
            const pv = getW(node, "prompt_view");
            if (pv) pv.value = output.prompt_view[0];
            app.graph.setDirtyCanvas(true, true);
        }
    }
});

console.log("[LLMPromptStudio.Bridge] extension loaded");
