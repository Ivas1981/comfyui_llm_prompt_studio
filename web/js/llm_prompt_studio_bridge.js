import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import {
    getW,
    isWriter, isCritic, isSmartSave, isLoader, isScene, isSmartLoader,
    lastSaveData, loopCounters,
} from "./llm_prompt_studio_shared.js";

import {
    refreshModels, saveToLibrary, refreshScenes, sendToWriter, requeuePrompt,
} from "./llm_prompt_studio_actions.js";

// ---------------------------------------------------------------------------
// Button registration
// ---------------------------------------------------------------------------
function addButton(node, label, handler) {
    if (node.widgets.some(w => w.name === label)) return;  // avoid duplicates
    node.addWidget("button", label, null, handler);
}

app.registerExtension({
    name: "llm_prompt_studio.bridge",

    // Make loading a saved workflow robust: ComfyUI validates each widget's saved value
    // against the combo options at load time, but the model list may not be populated yet
    // (it is fetched asynchronously). Inject the saved model value into the combo's allowed
    // options *before* ComfyUI validates, so a saved workflow never trips "Value not in list"
    // even when the server list hasn't arrived.
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!["LLMPromptStudioWriter", "LLMPromptStudioCritic",
              "LLMPromptStudioSceneBuilder"].includes(nodeData.name)) {
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
                }
            }
            return configure.apply(this, arguments);
        };
    },

    nodeCreated(node) {
        if (isWriter(node) || isCritic(node)) {
            addButton(node, "🔄 Refresh models", () => refreshModels(node));
        }
        if (isScene(node)) {
            addButton(node, "→ Send to Writer", () => sendToWriter(node));
            // Scene Builder also has a model combo, so give it the same manual refresh.
            addButton(node, "🔄 Refresh models", () => refreshModels(node));
        }
        if (isSmartSave(node)) {
            addButton(node, "💾 Save prompt to library", () => saveToLibrary(node));
        }
        if (isLoader(node)) {
            addButton(node, "🔄 Refresh scene list", () => refreshScenes(node));
        }
        // Auto-populate the model list on creation so a saved workflow whose model is not
        // yet in the combo validates without requiring a manual Refresh first. Deferred so it
        // runs after the graph is fully built.
        if (isWriter(node) || isCritic(node) || isScene(node)) {
            setTimeout(() => refreshModels(node), 400);
        }
    },
});

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
            node.title = "LLM Prompt Studio Image Critic — score " + score +
                         (approved ? "  ✓ approved" : "  ✗ rejected");
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
                const writer = app.graph.nodes.find(isWriter);
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

    // --- Smart Loader: detected family in the title ---
    if (isSmartLoader(node)) {
        const family = output.family ? output.family[0] : "";
        if (family) {
            node.title = "LLM Prompt Studio Smart Loader — family: " + family;
            app.graph.setDirtyCanvas(true, true);
        }
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