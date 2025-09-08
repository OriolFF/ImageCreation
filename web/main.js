const $ = (sel) => document.querySelector(sel);

// Target the running FastAPI backend
const API_BASE = "http://localhost:8000";
const API_URL = `${API_BASE}/v1/images/generations`;
const MODELS_URL = `${API_BASE}/v1/models`;
const SELECT_MODEL_URL = `${API_BASE}/v1/models/select`;

let ctrl = null;
let lastModelsInfo = null; // store last /v1/models payload

function setBusy(isBusy) {
  if (isBusy) {
    $("#progress").classList.remove("hidden");
    $("#generateBtn").disabled = true;
    $("#cancelBtn").disabled = false;
    $("#error").classList.add("hidden");
    $("#prompt").disabled = true;
    const ms = $("#modelSelect");
    if (ms) ms.disabled = true;
  } else {
    $("#progress").classList.add("hidden");
    $("#generateBtn").disabled = false;
    $("#cancelBtn").disabled = true;
    $("#prompt").disabled = false;
    const ms = $("#modelSelect");
    if (ms) ms.disabled = false;
    // Return focus to the input for faster next prompt entry
    $("#prompt").focus();
  }
}

// Lightweight busy mode for model switching only (do not show image-generation progress)
function setModelBusy(isBusy) {
  const ms = $("#modelSelect");
  const badge = $("#runtimeInfo");
  const genBtn = $("#generateBtn");
  const prompt = $("#prompt");
  if (isBusy) {
    if (ms) ms.disabled = true;
    if (badge) badge.textContent = "Switching model…";
    if (genBtn) genBtn.disabled = true;
    if (prompt) prompt.disabled = true;
  } else {
    if (ms) ms.disabled = false;
    // badge will be refreshed by loadModels()
    if (genBtn) genBtn.disabled = false;
    if (prompt) prompt.disabled = false;
  }
}

// --- Presets and model-aware configuration ---
const DEFAULT_PRESETS = {
  low:    { height: 512, width: 512, steps: 4,  guidance: 3.0 },
  medium: { height: 512, width: 512, steps: 12, guidance: 3.0 },
  high:   { height: 640, width: 640, steps: 16, guidance: 3.5 },
};

// For FLUX.1-dev: higher resolutions and >= 24 steps
const DEV_PRESETS = {
  low:    { height: 640, width: 640, steps: 24, guidance: 3.0 },
  medium: { height: 704, width: 704, steps: 28, guidance: 3.0 },
  high:   { height: 768, width: 768, steps: 32, guidance: 3.0 },
};

let currentPresetMap = { ...DEFAULT_PRESETS };

function updatePresetsForModel(modelKey) {
  const presetSelect = document.querySelector('#presetSelect');
  if (!presetSelect) return;
  const isDev = modelKey === 'dev';
  currentPresetMap = isDev ? { ...DEV_PRESETS } : { ...DEFAULT_PRESETS };

  // Update option labels to reflect current preset values
  const labelFor = (p) => `${p.height}×${p.width}, ${p.steps} steps`;
  const options = Array.from(presetSelect.options);
  const keys = ['low', 'medium', 'high']; // keep 'custom' untouched
  keys.forEach((k) => {
    const opt = options.find(o => o.value === k);
    if (opt) opt.textContent = `${k[0].toUpperCase()}${k.slice(1)} (${labelFor(currentPresetMap[k])})`;
  });

  // When switching to 'dev', default to a sensible preset automatically (unless user chose custom)
  if (isDev && presetSelect.value !== 'custom') {
    presetSelect.value = 'medium'; // 704×704, 28 steps for dev
    presetSelect.dispatchEvent(new Event('change'));
  }
}

function roundTo64(n) {
  const x = Math.max(256, parseInt(n || 0, 10));
  return Math.round(x / 64) * 64;
}

async function loadModels() {
  try {
    console.log("[UI] loadModels: fetching", MODELS_URL);
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(MODELS_URL, { signal: controller.signal });
    clearTimeout(t);
    console.log("[UI] loadModels: response status", res.status);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    console.log("[UI] loadModels: parsed JSON", json);
    lastModelsInfo = json;
    const select = $("#modelSelect");
    const badge = $("#runtimeInfo");
    // Clear any existing options
    select.innerHTML = "";
    // Populate with available models
    const available = json.available || {};
    const entries = Object.entries(available); // [key, id]
    for (const [key, id] of entries) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = `${key} (${id})`;
      select.appendChild(opt);
    }
    // Add custom option if active model not in registry
    const active = json.active || {};
    if (active.key && !available[active.key]) {
      const opt = document.createElement("option");
      opt.value = active.key;
      opt.textContent = `${active.key} (${active.id})`;
      select.appendChild(opt);
    }
    // Set current value and adapt presets for the active model
    if (active.key) {
      select.value = active.key;
      updatePresetsForModel(active.key);
    }
    // Update runtime info badge (active model id + device/dtype)
    if (badge) {
      const device = json.device || "";
      const dtype = json.dtype || "";
      const activeId = active.id || "";
      badge.textContent = `${activeId} | ${device} | ${dtype}`;
    }
  } catch (err) {
    console.error("[UI] loadModels: failed", err);
    showError(`Failed to load models: ${err.message}`);
    // Show a clear fallback option in the selector
    const select = $("#modelSelect");
    if (select) {
      select.innerHTML = "";
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Models unavailable";
      select.appendChild(opt);
    }
    const badge = $("#runtimeInfo");
    if (badge) badge.textContent = "backend offline?";
  }
}

async function selectModel(keyOrRepo) {
  try {
    console.log("[UI] selectModel: switching to", keyOrRepo);
    setModelBusy(true);
    const res = await fetch(SELECT_MODEL_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: keyOrRepo }),
    });
    console.log("[UI] selectModel: response status", res.status);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${text}`);
    }
    // Reload models to reflect active selection and any custom mapping
    await loadModels();
    console.log("[UI] selectModel: switched to", keyOrRepo);
  } catch (err) {
    console.error("[UI] selectModel: failed", err);
    showError(`Failed to switch model: ${err.message}`);
  } finally {
    setModelBusy(false);
  }
}

// Initialize model selector
document.addEventListener("DOMContentLoaded", () => {
  console.log("[UI] DOMContentLoaded: initializing UI and loading models...");
  loadModels();
  const select = $("#modelSelect");
  if (select) {
    select.addEventListener("change", (e) => {
      const value = e.target.value;
      if (value) selectModel(value);
    });
  }

  // Preset/custom toggling
  const presetSelect = $("#presetSelect");
  const customRow = $("#customRow");
  const heightInput = $("#heightInput");
  const widthInput = $("#widthInput");
  const stepsInput = $("#stepsInput");
  const guidanceInput = $("#guidanceInput");
  if (presetSelect) {
    const toggleCustom = () => {
      const isCustom = presetSelect.value === "custom";
      [heightInput, widthInput, stepsInput, guidanceInput].forEach((el) => {
        if (!el) return;
        el.disabled = !isCustom;
      });
      if (customRow) customRow.style.opacity = isCustom ? "1" : "0.6";
    };
    presetSelect.addEventListener("change", toggleCustom);
    toggleCustom();
  }
  const cacheLink = $("#cacheInfo");
  if (cacheLink) {
    cacheLink.addEventListener("click", (e) => {
      e.preventDefault();
      const info = lastModelsInfo || {};
      const cache = info.cache_dir || "(default)";
      const rev = info.revision || "(latest)";
      const variant = info.variant || "(default)";
      const device = info.device || "";
      const dtype = info.dtype || "";
      const active = info.active?.id || "";
      alert(`Active: ${active}\nDevice: ${device}\nDType: ${dtype}\nCache: ${cache}\nRevision: ${rev}\nVariant: ${variant}`);
    });
  }
  console.log("[UI] DOMContentLoaded: event handlers registered.");
});

function showError(message) {
  const el = $("#error");
  el.textContent = message || "Something went wrong.";
  el.classList.remove("hidden");
}

function showNotice(message) {
  const el = $("#notice");
  if (!el) return;
  if (message) {
    el.textContent = message;
    el.classList.remove("hidden");
  } else {
    el.textContent = "";
    el.classList.add("hidden");
  }
}

function appendMessage({ role, text, imageSrc, savedPath }) {
  const chat = $("#chat");
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (text) bubble.textContent = text;

  wrapper.appendChild(roleEl);
  wrapper.appendChild(bubble);

  if (imageSrc) {
    const frame = document.createElement("div");
    frame.className = "image-frame";
    const img = document.createElement("img");
    img.alt = "Generated image";
    img.src = imageSrc;
    img.style.maxWidth = "100%";
    img.style.borderRadius = "8px";
    frame.appendChild(img);

    const actions = document.createElement("div");
    actions.style.marginTop = "8px";
    const link = document.createElement("a");
    link.className = "text-button";
    link.textContent = "Download";
    link.download = "output.png";
    link.href = imageSrc;
    actions.appendChild(link);

    if (savedPath) {
      const hint = document.createElement("div");
      hint.className = "caption";
      hint.textContent = `Saved to ${savedPath}`;
      actions.appendChild(hint);
    }

    wrapper.appendChild(frame);
    wrapper.appendChild(actions);
  }

  chat.appendChild(wrapper);
  chat.scrollTop = chat.scrollHeight;
}

async function generate() {
  console.log("[UI] generate: starting request...");
  const prompt = $("#prompt").value.trim();
  const storeLocal = $("#storeLocal").checked;
  const preset = $("#presetSelect")?.value || "medium";
  if (!prompt) {
    showError("Please enter a prompt.");
    return;
  }

  // Add user message to chat
  appendMessage({ role: "user", text: prompt });

  // Clear input immediately so it doesn't appear duplicated while generating
  $("#prompt").value = "";

  ctrl = new AbortController();
  setBusy(true);

  try {
    // Determine presets by current model
    const activeKey = lastModelsInfo?.active?.key || "schnell";
    updatePresetsForModel(activeKey);
    let p;
    if (preset === "custom") {
      const h = roundTo64($("#heightInput")?.value);
      const w = roundTo64($("#widthInput")?.value);
      const s = Math.max(1, parseInt($("#stepsInput")?.value || 12, 10));
      const g = parseFloat($("#guidanceInput")?.value || 3.0);
      p = { height: h, width: w, num_inference_steps: s, guidance_scale: g };
    } else {
      const base = currentPresetMap[preset] || currentPresetMap.medium;
      p = { height: base.height, width: base.width, num_inference_steps: base.steps, guidance_scale: base.guidance };
    }

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, store_local: storeLocal, ...p }),
      signal: ctrl.signal,
    });
    console.log("[UI] generate: response status", res.status);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${res.statusText} ${text}`);
    }

    const json = await res.json();
    console.log("[UI] generate: parsed JSON");
    const item = json?.data?.[0];
    const b64 = item?.b64_json;
    const savedPath = item?.saved_path;
    const usedModel = item?.model;
    const usedModelKey = item?.model_key;
    const params = item?.params || {};
    if (!b64) throw new Error("Invalid response payload");

    const src = `data:image/png;base64,${b64}`;
    // Show OOM fallback details if present
    if (params.fallback_applied) {
      const orig = params.original_params || {};
      showNotice(`Memory fallback: requested ${orig.width}x${orig.height}, ${orig.num_inference_steps} steps → used ${params.width}x${params.height}, ${params.num_inference_steps} steps.`);
    } else {
      showNotice("");
    }

    // Include a small hint about which model generated this image
    if (savedPath || usedModel || usedModelKey) {
      const hintParts = [];
      if (savedPath) hintParts.push(`Saved to ${savedPath}`);
      if (usedModel || usedModelKey) hintParts.push(`Model: ${usedModelKey || ""} ${usedModel ? `(${usedModel})` : ""}`.trim());
      appendMessage({ role: "assistant", text: "Here is your image:", imageSrc: src, savedPath: hintParts.join(" • ") });
    } else {
      appendMessage({ role: "assistant", text: "Here is your image:", imageSrc: src, savedPath });
    }
    $("#prompt").value = "";
  } catch (err) {
    if (err.name === "AbortError") return; // user canceled
    console.error("[UI] generate: failed", err);
    showError(err.message);
  } finally {
    setBusy(false);
    ctrl = null;
    console.log("[UI] generate: done.");
  }
}

function cancel() {
  if (ctrl) ctrl.abort();
}

$("#generateBtn").addEventListener("click", generate);
$("#cancelBtn").addEventListener("click", cancel);
$("#prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    generate();
  }
});
