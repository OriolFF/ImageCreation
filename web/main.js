const $ = (sel) => document.querySelector(sel);

// Target the running FastAPI backend
const API_URL = "http://localhost:8000/v1/images/generations";

let ctrl = null;

function setBusy(isBusy) {
  if (isBusy) {
    $("#progress").classList.remove("hidden");
    $("#generateBtn").disabled = true;
    $("#cancelBtn").disabled = false;
    $("#error").classList.add("hidden");
    $("#prompt").disabled = true;
  } else {
    $("#progress").classList.add("hidden");
    $("#generateBtn").disabled = false;
    $("#cancelBtn").disabled = true;
    $("#prompt").disabled = false;
    // Return focus to the input for faster next prompt entry
    $("#prompt").focus();
  }
}

function showError(message) {
  const el = $("#error");
  el.textContent = message || "Something went wrong.";
  el.classList.remove("hidden");
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
  const prompt = $("#prompt").value.trim();
  const storeLocal = $("#storeLocal").checked;
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
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, store_local: storeLocal }),
      signal: ctrl.signal,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${res.statusText} ${text}`);
    }

    const json = await res.json();
    const item = json?.data?.[0];
    const b64 = item?.b64_json;
    const savedPath = item?.saved_path;
    if (!b64) throw new Error("Invalid response payload");

    const src = `data:image/png;base64,${b64}`;
    appendMessage({ role: "assistant", text: "Here is your image:", imageSrc: src, savedPath });

    // reset composer for next turn
    $("#prompt").value = "";
  } catch (err) {
    if (err.name === "AbortError") return; // user canceled
    console.error(err);
    showError(err.message);
  } finally {
    setBusy(false);
    ctrl = null;
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
