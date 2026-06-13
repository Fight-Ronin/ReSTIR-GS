const viewport = document.getElementById("viewport");
const viewportWrap = document.getElementById("viewport-wrap");
const busyBadge = document.getElementById("busy");
const metrics = document.getElementById("metrics");
const statusBox = document.getElementById("status");
const assetTitle = document.getElementById("asset-title");
const viewBadge = document.getElementById("view-badge");
const frameBadge = document.getElementById("frame-badge");
const viewportReadout = document.getElementById("viewport-readout");
const viewOverlay = document.getElementById("view-overlay");
const viewMode = document.getElementById("view-mode");
const viewDetail = document.getElementById("view-detail");
const interactionFeedback = document.getElementById("interaction-feedback");
const layerButtons = Array.from(document.querySelectorAll("[data-view]"));
const resetButton = document.getElementById("reset-button");
const saveButton = document.getElementById("save-button");

let snapshot = null;
let busy = false;
let drag = null;
let pendingControlDown = false;
let resizeTimer = null;
let controlHoldTimer = null;
let pendingDragAction = null;
let pendingOneShotAction = null;
let resizePending = false;
let movementCursor = 0;
const activeMovementKeys = new Set();

const viewLabels = {
  rgb: "RGB",
  alpha: "Alpha",
  depth: "Depth",
  normal: "Normal",
  lambertian: "Lambertian",
  blinn_phong: "Blinn-Phong",
};
const viewProfiles = {
  rgb: { mode: "RGB", detail: "composite", overlay: false },
  alpha: { mode: "Alpha", detail: "0..1 mask", overlay: true },
  depth: { mode: "Depth", detail: "near -> far", overlay: true },
  normal: { mode: "Normal", detail: "camera xyz", overlay: true },
  lambertian: { mode: "Lambertian", detail: "diffuse RIS", overlay: false },
  blinn_phong: { mode: "Blinn-Phong", detail: "shader", overlay: false },
};
const minRenderDimension = 64;
const maxRenderDimension = 2048;
const controlHoldDelayMs = 160;
const movementMap = {
  w: "forward",
  s: "backward",
  a: "left",
  d: "right",
  shift: "up",
  control: "down",
};
const movementOrder = ["w", "s", "a", "d", "shift", "control"];
const inputLabels = {
  w: "W",
  s: "S",
  a: "A",
  d: "D",
  shift: "Shift",
  control: "Ctrl",
  orbit: "Orbit",
  look: "Look",
  pan: "Pan",
  resize: "Resize",
  render: "Render",
};

function setBusy(value) {
  busy = value;
  busyBadge.classList.toggle("hidden", !value);
  setInteractionFeedback();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function updateUi(data) {
  snapshot = data;
  const asset = data.asset || {};
  const render = data.render || {};
  const camera = data.camera || {};
  const viewLabel = viewLabels[data.view] || data.view;
  const frameNumber = Number(data.frame_index) + 1;
  document.documentElement.dataset.view = data.view || "rgb";
  assetTitle.textContent = asset.label || "ReSTIR-GS Viewer";
  setMetricChips([
    `frame ${frameNumber}`,
    `${viewLabel}`,
    `${Number(render.render_ms || 0).toFixed(1)} ms`,
    `${Number(render.valid_pixels || 0)} px`,
    `yaw ${Number(camera.yaw_degrees || 0).toFixed(1)}`,
    `pitch ${Number(camera.pitch_degrees || 0).toFixed(1)}`,
    `radius ${Number(camera.radius || 0).toFixed(3)}`,
  ]);
  viewBadge.textContent = viewLabel;
  frameBadge.textContent = `Frame ${frameNumber}`;
  updateViewOverlay(data.view);
  viewportReadout.textContent = [
    `${Number(camera.width || 0)} x ${Number(camera.height || 0)}`,
    `${Number(render.render_ms || 0).toFixed(1)} ms`,
    data.status || "ready",
  ].join("  /  ");
  statusBox.textContent = data.status || "ready";
  layerButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.view === data.view);
  });
  setInteractionFeedback();
}

function updateViewOverlay(view) {
  const profile = viewProfiles[view] || { mode: viewLabels[view] || view, detail: "", overlay: true };
  viewMode.textContent = profile.mode;
  viewDetail.textContent = profile.detail;
  viewOverlay.classList.toggle("hidden", !profile.overlay);
}

function setInteractionFeedback() {
  const chips = [];
  if (busy) {
    chips.push({ label: inputLabels.render, accent: true });
  }
  if (drag) {
    chips.push({ label: inputLabels[drag.mode] || drag.mode, accent: true });
  }
  activeMovementKeys.forEach((key) => {
    chips.push({ label: inputLabels[key] || key.toUpperCase(), accent: true });
  });
  if (pendingControlDown) {
    chips.push({ label: inputLabels.control, accent: false });
  }
  if (pendingDragAction !== null) {
    chips.push({ label: `${inputLabels[pendingDragAction.action] || pendingDragAction.action} queued`, accent: false });
  }
  if (pendingOneShotAction !== null) {
    chips.push({ label: "Action queued", accent: false });
  }
  if (resizePending) {
    chips.push({ label: `${inputLabels.resize} queued`, accent: false });
  }
  interactionFeedback.classList.toggle("hidden", chips.length === 0);
  interactionFeedback.replaceChildren(
    ...chips.map((chip) => {
      const element = document.createElement("span");
      element.className = chip.accent ? "input-chip accent" : "input-chip";
      element.textContent = chip.label;
      return element;
    }),
  );
}

function setMetricChips(values) {
  metrics.replaceChildren(
    ...values.map((value) => {
      const chip = document.createElement("span");
      chip.className = "metric-chip";
      chip.textContent = value;
      return chip;
    }),
  );
}

function refreshImage() {
  viewport.src = `/api/image.png?ts=${Date.now()}`;
}

async function refreshSnapshot() {
  const data = await fetchJson("/api/snapshot");
  updateUi(data);
  refreshImage();
}

async function postJson(url, payload = null) {
  if (busy) {
    return false;
  }
  setBusy(true);
  try {
    const options = { method: "POST" };
    if (payload !== null) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(payload);
    }
    const data = await fetchJson(url, options);
    updateUi(data.snapshot || data);
    refreshImage();
    return true;
  } catch (error) {
    statusBox.textContent = `error: ${error.message}`;
    return false;
  } finally {
    setBusy(false);
    drainRenderQueue();
  }
}

function action(payload) {
  return enqueueOneShotAction(payload);
}

function setView(view) {
  return postJson("/api/view", { view });
}

function saveCurrent() {
  return postJson("/api/save");
}

function viewportRenderSize() {
  const rect = viewportWrap.getBoundingClientRect();
  const width = Math.round(Math.max(minRenderDimension, Math.min(rect.width, maxRenderDimension)));
  const height = Math.round(Math.max(minRenderDimension, Math.min(rect.height, maxRenderDimension)));
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    return null;
  }
  return { width, height };
}

function scheduleViewportResize() {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    resizePending = true;
    setInteractionFeedback();
    drainRenderQueue();
  }, 180);
}

function nextResizeRequest() {
  if (!resizePending || drag !== null || pendingDragAction !== null || activeMovementKeys.size > 0 || pendingOneShotAction !== null) {
    return null;
  }
  const size = viewportRenderSize();
  if (size === null) {
    resizePending = false;
    setInteractionFeedback();
    return null;
  }
  const camera = (snapshot && snapshot.camera) || {};
  if (Number(camera.width) === size.width && Number(camera.height) === size.height) {
    resizePending = false;
    setInteractionFeedback();
    return null;
  }
  resizePending = false;
  setInteractionFeedback();
  return { url: "/api/resize", payload: size };
}

function enqueueDragAction(mode, dx, dy) {
  if (pendingDragAction && pendingDragAction.action === mode) {
    pendingDragAction.dx += dx;
    pendingDragAction.dy += dy;
  } else {
    pendingDragAction = { action: mode, dx, dy };
  }
  setInteractionFeedback();
  drainRenderQueue();
}

function enqueueOneShotAction(payload) {
  if (busy || pendingDragAction !== null || activeMovementKeys.size > 0) {
    pendingOneShotAction = payload;
    setInteractionFeedback();
    drainRenderQueue();
    return Promise.resolve(false);
  }
  return postJson("/api/action", payload);
}

function nextMovementCommand() {
  if (activeMovementKeys.size === 0) {
    return null;
  }
  for (let offset = 0; offset < movementOrder.length; offset += 1) {
    const index = (movementCursor + offset) % movementOrder.length;
    const key = movementOrder[index];
    if (activeMovementKeys.has(key)) {
      movementCursor = (index + 1) % movementOrder.length;
      return movementMap[key];
    }
  }
  return null;
}

function nextQueuedRequest() {
  if (pendingDragAction !== null) {
    const payload = pendingDragAction;
    pendingDragAction = null;
    setInteractionFeedback();
    return { url: "/api/action", payload };
  }
  if (pendingOneShotAction !== null) {
    const payload = pendingOneShotAction;
    pendingOneShotAction = null;
    setInteractionFeedback();
    return { url: "/api/action", payload };
  }
  const movementCommand = nextMovementCommand();
  if (movementCommand !== null) {
    return { url: "/api/action", payload: { action: "move", command: movementCommand } };
  }
  return nextResizeRequest();
}

function drainRenderQueue() {
  if (busy) {
    return;
  }
  const request = nextQueuedRequest();
  if (request !== null) {
    void postJson(request.url, request.payload);
  }
}

function startMovementKey(key) {
  if (key === "control") {
    if (pendingControlDown || activeMovementKeys.has("control")) {
      return;
    }
    pendingControlDown = true;
    window.clearTimeout(controlHoldTimer);
    controlHoldTimer = window.setTimeout(() => {
      if (pendingControlDown) {
        pendingControlDown = false;
        activeMovementKeys.add("control");
        setInteractionFeedback();
        drainRenderQueue();
      }
    }, controlHoldDelayMs);
    setInteractionFeedback();
    return;
  }
  activeMovementKeys.add(key);
  setInteractionFeedback();
  drainRenderQueue();
}

function stopMovementKey(key) {
  if (key === "control") {
    window.clearTimeout(controlHoldTimer);
    if (pendingControlDown) {
      pendingControlDown = false;
      enqueueOneShotAction({ action: "move", command: "down" });
    } else {
      activeMovementKeys.delete("control");
      setInteractionFeedback();
      drainRenderQueue();
    }
    return;
  }
  activeMovementKeys.delete(key);
  setInteractionFeedback();
  drainRenderQueue();
}

layerButtons.forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

resetButton.addEventListener("click", () => action({ action: "reset" }));
saveButton.addEventListener("click", () => saveCurrent());

document.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && key === "s") {
    pendingControlDown = false;
    activeMovementKeys.delete("control");
    window.clearTimeout(controlHoldTimer);
    setInteractionFeedback();
    event.preventDefault();
    saveCurrent();
    return;
  }
  const numberMap = {
    "1": "rgb",
    "2": "alpha",
    "3": "depth",
    "4": "normal",
    "5": "lambertian",
    "6": "blinn_phong",
  };
  if (movementMap[key]) {
    event.preventDefault();
    startMovementKey(key);
    return;
  }
  if (event.repeat || busy) {
    return;
  }
  if (numberMap[key]) {
    event.preventDefault();
    setView(numberMap[key]);
    return;
  }
  if (key === "[") {
    event.preventDefault();
    action({ action: "frame", delta: -1 });
    return;
  }
  if (key === "]") {
    event.preventDefault();
    action({ action: "frame", delta: 1 });
    return;
  }
  if (key === "r") {
    event.preventDefault();
    action({ action: "reset" });
  }
});

document.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  if (movementMap[key]) {
    event.preventDefault();
    stopMovementKey(key);
  }
});

viewportWrap.addEventListener("pointerdown", (event) => {
  let mode = null;
  if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
    mode = "pan";
  } else if (event.button === 2) {
    mode = "look";
  } else if (event.button === 0) {
    mode = "orbit";
  }
  if (mode === null) {
    return;
  }
  event.preventDefault();
  drag = {
    x: event.clientX,
    y: event.clientY,
    mode,
  };
  setInteractionFeedback();
  viewportWrap.setPointerCapture(event.pointerId);
});

viewportWrap.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});

viewportWrap.addEventListener("pointermove", (event) => {
  if (!drag) {
    return;
  }
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  if (Math.abs(dx) < 1 && Math.abs(dy) < 1) {
    return;
  }
  drag.x = event.clientX;
  drag.y = event.clientY;
  enqueueDragAction(drag.mode, dx, dy);
});

viewportWrap.addEventListener("pointerup", () => {
  drag = null;
  setInteractionFeedback();
  drainRenderQueue();
});

viewportWrap.addEventListener("pointerleave", () => {
  drag = null;
  setInteractionFeedback();
  drainRenderQueue();
});

viewportWrap.addEventListener("lostpointercapture", () => {
  drag = null;
  setInteractionFeedback();
  drainRenderQueue();
});

window.addEventListener("blur", () => {
  pendingControlDown = false;
  activeMovementKeys.clear();
  window.clearTimeout(controlHoldTimer);
  drag = null;
  setInteractionFeedback();
});

viewportWrap.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    action({ action: "dolly", scale: event.deltaY < 0 ? 0.9 : 1.1 });
  },
  { passive: false },
);

refreshSnapshot().catch((error) => {
  statusBox.textContent = `error: ${error.message}`;
}).finally(() => {
  scheduleViewportResize();
});

if ("ResizeObserver" in window) {
  const resizeObserver = new ResizeObserver(() => scheduleViewportResize());
  resizeObserver.observe(viewportWrap);
} else {
  window.addEventListener("resize", scheduleViewportResize);
}
