// ComfyUI-H3-SeedScout — interactive seed picker widget.
// Dependency-free. Uses the window.comfyAPI globals, the same idiom the other packs
// in this install use (see ComfyUI-KJNodes/web/js/fast_preview_batch.js), with a
// dynamic-import fallback for older/newer frontends.
// Frontend targeted: comfyui-frontend-package 1.47.12.

const NODE_CLASS = "MiniMaxH3SeedScoutSampler";
const EVT_PREVIEW = "h3_seed_scout_preview";
const EVT_WAITING = "h3_seed_scout_waiting";
const EVT_DONE = "h3_seed_scout_done";
const SELECT_ROUTE = "/h3_seed_scout/select";
const STYLE_ID = "h3-seed-scout-style";

// Widgets kept visible on the node; everything else is hidden for a clean look.
// Hidden widgets keep their values (defaults or whatever the loaded workflow set)
// and stay fully scriptable via the API JSON.
const VISIBLE_WIDGETS = new Set([
    "seed_start", "seed_count", "scout_step", "selection_timeout",
]);

function hideExtraWidgets(node) {
    if (!node.widgets) return;
    for (const w of node.widgets) {
        if (w.name === "h3_seed_scout_ui" || VISIBLE_WIDGETS.has(w.name)) continue;
        // standard ComfyUI hidden-widget idiom (same trick rgthree/KJNodes use)
        w.origType = w.type;
        w.type = "hidden";
        w.hidden = true;
        w.computeSize = () => [0, -4];
    }
}

async function getApis() {
    const g = window.comfyAPI;
    if (g && g.app && g.api) return { app: g.app.app, api: g.api.api };
    const appMod = await import("../../scripts/app.js");
    const apiMod = await import("../../scripts/api.js");
    return { app: appMod.app, api: apiMod.api };
}

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.h3ss-root { display:flex; flex-direction:column; gap:6px; padding:6px;
    font-family: sans-serif; font-size:12px; color:#ddd; box-sizing:border-box; width:100%; }
.h3ss-status { min-height:16px; opacity:0.85; }
.h3ss-status.waiting { color:#ffd479; font-weight:600; }
.h3ss-status.done { color:#8fe388; }
.h3ss-status.error { color:#ff8f8f; }
.h3ss-stage { display:flex; align-items:center; justify-content:center;
    background:#181818; border:1px solid #333; border-radius:4px; min-height:120px;
    overflow:hidden; }
.h3ss-stage img { max-width:100%; max-height:320px; display:block; image-rendering:auto; }
.h3ss-empty { opacity:0.5; padding:24px 8px; }
.h3ss-seeds { display:flex; flex-wrap:wrap; gap:4px; }
.h3ss-seed { flex:1 1 auto; min-width:34px; padding:4px 6px; cursor:pointer;
    background:#2b2b2b; border:1px solid #444; border-radius:4px; color:#ddd;
    font-size:12px; line-height:1.1; user-select:none; }
.h3ss-seed:hover { background:#3a3a3a; }
.h3ss-seed.sel { background:#3b5b8c; border-color:#7aa7e0; color:#fff; font-weight:600; }
.h3ss-seed[disabled] { opacity:0.45; cursor:default; }
.h3ss-seed .sub { display:block; font-size:9px; opacity:0.65; font-weight:400; }
.h3ss-row { display:flex; gap:4px; align-items:center; }
.h3ss-continue { flex:1; padding:6px; cursor:pointer; border-radius:4px;
    background:#2f6f3f; border:1px solid #4d9c60; color:#fff; font-weight:600; }
.h3ss-continue:hover { background:#3a8850; }
.h3ss-continue[disabled] { opacity:0.4; cursor:default; }
.h3ss-hint { opacity:0.55; font-size:10px; }
.h3ss-gear { flex:0 0 auto; padding:6px 10px; cursor:pointer; border-radius:4px;
    background:#2b2b2b; border:1px solid #444; color:#ddd; }
.h3ss-gear:hover { background:#3a3a3a; }
.h3ss-settings { display:none; flex-direction:column; gap:4px; padding:6px;
    background:#202020; border:1px solid #3a3a3a; border-radius:4px; }
.h3ss-settings.open { display:flex; }
.h3ss-set-row { display:flex; align-items:center; gap:6px; }
.h3ss-set-row label { flex:1; opacity:0.8; font-size:11px; }
.h3ss-set-row input, .h3ss-set-row select { flex:1; background:#181818;
    border:1px solid #444; border-radius:3px; color:#ddd; font-size:11px;
    padding:2px 4px; min-width:0; }
`;
    document.head.appendChild(s);
}

function makeUI(node, api) {
    ensureStyles();

    const root = document.createElement("div");
    root.className = "h3ss-root";

    const status = document.createElement("div");
    status.className = "h3ss-status";
    status.textContent = "idle";

    const stage = document.createElement("div");
    stage.className = "h3ss-stage";
    const empty = document.createElement("div");
    empty.className = "h3ss-empty";
    empty.textContent = "no previews yet — queue this prompt in interactive mode";
    stage.appendChild(empty);
    const img = document.createElement("img");
    img.style.display = "none";
    stage.appendChild(img);

    const seedsRow = document.createElement("div");
    seedsRow.className = "h3ss-seeds";

    const btnRow = document.createElement("div");
    btnRow.className = "h3ss-row";
    const cont = document.createElement("button");
    cont.className = "h3ss-continue";
    cont.textContent = "Continue ▶";
    cont.disabled = true;
    const gear = document.createElement("button");
    gear.className = "h3ss-gear";
    gear.textContent = "⚙";
    gear.title = "advanced settings";
    btnRow.append(cont, gear);

    // advanced-settings popup: edits the node's hidden widgets in place
    const settings = document.createElement("div");
    settings.className = "h3ss-settings";
    function buildSettings() {
        settings.innerHTML = "";
        for (const w of node.widgets || []) {
            if (!w.hidden || w.name === "h3_seed_scout_ui") continue;
            if (w.name === "control_after_generate") continue;
            const row = document.createElement("div");
            row.className = "h3ss-set-row";
            const label = document.createElement("label");
            label.textContent = w.name;
            let field;
            const opts = w.options && w.options.values;
            if (Array.isArray(opts)) {
                field = document.createElement("select");
                for (const o of opts) {
                    const el = document.createElement("option");
                    el.value = el.textContent = o;
                    if (o === w.value) el.selected = true;
                    field.appendChild(el);
                }
                field.addEventListener("change", () => { w.value = field.value; });
            } else {
                field = document.createElement("input");
                const numeric = typeof w.value === "number";
                field.type = numeric ? "number" : "text";
                field.value = w.value;
                field.addEventListener("change", () => {
                    w.value = numeric ? Number(field.value) : field.value;
                });
            }
            row.append(label, field);
            settings.appendChild(row);
        }
    }
    gear.addEventListener("click", () => {
        const open = settings.classList.toggle("open");
        if (open) buildSettings();
    });

    const hint = document.createElement("div");
    hint.className = "h3ss-hint";
    hint.textContent = "click a seed to preview · double-click or Continue to confirm";

    root.append(status, stage, seedsRow, btnRow, settings, hint);

    const state = {
        root, status, stage, img, empty, seedsRow, cont,
        previews: new Map(),   // seed -> {url, index, elapsed}
        buttons: new Map(),    // seed -> button
        selected: null,
        waiting: false,
        remaining: 0,
        submitting: false,
    };

    function setStatus(text, cls) {
        status.textContent = text;
        status.className = "h3ss-status" + (cls ? " " + cls : "");
    }
    state.setStatus = setStatus;

    function show(seed) {
        const p = state.previews.get(seed);
        if (!p) return;
        state.selected = seed;
        img.src = p.url;
        img.style.display = "block";
        empty.style.display = "none";
        for (const [s, b] of state.buttons) b.classList.toggle("sel", s === seed);
        const w = node.widgets && node.widgets.find((x) => x.name === "selected_seed");
        if (w) w.value = seed;
    }
    state.show = show;

    async function submit(seed) {
        if (!state.waiting || state.submitting) return;
        const msg = "Continue with seed " + seed + " for the remaining " +
            state.remaining + " steps?";
        if (!window.confirm(msg)) return;
        state.submitting = true;
        try {
            const resp = await api.fetchApi(SELECT_ROUTE, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: String(node.id), seed: seed }),
            });
            const data = await resp.json().catch(() => ({}));
            if (resp.ok && data.ok) {
                state.waiting = false;
                setEnabled(false);
                setStatus("continuing seed " + seed + "…", "done");
            } else {
                setStatus("selection rejected: " + (data.error || resp.status), "error");
                state.submitting = false;
            }
        } catch (e) {
            setStatus("selection failed: " + e, "error");
            state.submitting = false;
        }
    }
    state.submit = submit;

    function setEnabled(on) {
        // seed buttons always stay clickable for browsing previews;
        // only the confirm/continue path is gated on the waiting state
        cont.disabled = !on;
    }
    state.setEnabled = setEnabled;

    function addSeed(seed, url, index, elapsed) {
        state.previews.set(seed, { url, index, elapsed });
        let b = state.buttons.get(seed);
        if (!b) {
            b = document.createElement("button");
            b.className = "h3ss-seed";
            b.title = "seed " + seed;
            const label = document.createElement("span");
            label.textContent = "#" + (index + 1);
            const sub = document.createElement("span");
            sub.className = "sub";
            sub.textContent = String(seed);
            b.append(label, sub);
            b.addEventListener("click", () => show(seed));
            b.addEventListener("dblclick", () => { show(seed); submit(seed); });
            state.buttons.set(seed, b);
            seedsRow.appendChild(b);
        }
        // refresh the stage when a provisional preview is upgraded to the VAE one
        if (state.selected === seed) img.src = url;
        if (state.selected === null) show(seed);
    }
    state.addSeed = addSeed;

    function reset() {
        for (const p of state.previews.values()) {
            try { URL.revokeObjectURL(p.url); } catch (e) { /* data: urls */ }
        }
        state.previews.clear();
        state.buttons.clear();
        seedsRow.innerHTML = "";
        state.selected = null;
        state.waiting = false;
        state.submitting = false;
        img.style.display = "none";
        img.removeAttribute("src");
        empty.style.display = "";
        setEnabled(false);
        setStatus("scouting…");
    }
    state.reset = reset;

    cont.addEventListener("click", () => {
        if (state.selected === null) {
            setStatus("pick a seed first", "error");
            return;
        }
        submit(state.selected);
    });

    return state;
}

function findNode(app, nodeId) {
    if (!app || !app.graph) return null;
    const n = app.graph.getNodeById(Number(nodeId));
    if (n) return n;
    for (const cand of app.graph._nodes || []) {
        if (String(cand.id) === String(nodeId)) return cand;
    }
    return null;
}

(async () => {
    const { app, api } = await getApis();

    app.registerExtension({
        name: "H3SeedScout.Interactive",

        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== NODE_CLASS) return;

            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onCreated ? onCreated.apply(this, arguments) : undefined;
                try {
                    const state = makeUI(this, api);
                    this.__h3ss = state;
                    this.addDOMWidget("h3_seed_scout_ui", "div", state.root, {
                        serialize: false,
                        hideOnZoom: false,
                    });
                    hideExtraWidgets(this);
                    const sz = this.computeSize();
                    this.setSize([Math.max(sz[0], 340), Math.max(sz[1], 420)]);
                } catch (e) {
                    console.error("[H3SeedScout] widget init failed", e);
                }
                return r;
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (this.__h3ss) this.__h3ss.reset();
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };
        },
    });

    api.addEventListener("executing", (e) => {
        const nodeId = e.detail && (e.detail.node ?? e.detail);
        if (nodeId === null || nodeId === undefined) return;
        const node = findNode(app, nodeId);
        if (node && node.__h3ss && node.comfyClass === NODE_CLASS) node.__h3ss.reset();
    });

    api.addEventListener(EVT_PREVIEW, (e) => {
        const d = e.detail || {};
        const node = findNode(app, d.node_id);
        if (!node || !node.__h3ss) return;
        const url = "data:" + (d.mime || "image/webp") + ";base64," + d.image_b64;
        node.__h3ss.addSeed(d.seed, url, d.index || 0, d.elapsed);
        node.__h3ss.setStatus(
            "scouted " + (Number(d.index) + 1) + "/" + (d.total || "?") + " seeds…");
        app.graph.setDirtyCanvas(true, false);
    });

    api.addEventListener(EVT_WAITING, (e) => {
        const d = e.detail || {};
        const node = findNode(app, d.node_id);
        if (!node || !node.__h3ss) return;
        const s = node.__h3ss;
        s.waiting = true;
        s.remaining = d.remaining_steps;
        s.setEnabled(true);
        s.setStatus(
            "waiting for selection — " + (d.seeds || []).length + " seeds, " +
            d.remaining_steps + " steps left" +
            (d.timeout ? " (timeout " + d.timeout + "s)" : ""),
            "waiting");
        app.graph.setDirtyCanvas(true, false);
    });

    api.addEventListener(EVT_DONE, (e) => {
        const d = e.detail || {};
        const node = findNode(app, d.node_id);
        if (!node || !node.__h3ss) return;
        const s = node.__h3ss;
        s.waiting = false;
        s.submitting = false;
        s.setEnabled(false);
        if (d.status === "continued") s.setStatus("continuing seed " + d.seed + "…", "done");
        else if (d.status === "timeout") s.setStatus("timed out — using seed " + d.seed, "error");
        else if (d.status === "cancelled") s.setStatus("cancelled", "error");
        else s.setStatus("interrupted", "error");
        app.graph.setDirtyCanvas(true, false);
    });

    console.log("[H3SeedScout] interactive extension loaded");
})();
