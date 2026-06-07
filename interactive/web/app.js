const viewport = document.getElementById("viewport");
const viewportWrap = document.getElementById("viewport-wrap");
const busyBadge = document.getElementById("busy");
const metrics = document.getElementById("metrics");
const statusBox = document.getElementById("status");
const assetTitle = document.getElementById("asset-title");
const viewBadge = document.getElementById("view-badge");
const frameBadge = document.getElementById("frame-badge");
const viewportReadout = document.getElementById("viewport-readout");
const layerButtons = Array.from(document.querySelectorAll("[data-view]"));
const resetButton = document.getElementById("reset-button");
const saveButton = document.getElementById("save-button");

let snapshot = null;
let busy = false;
let drag = null;
let pendingControlDown = false;
let resizeTimer = null;

const viewLabels = {
  rgb: "RGB",
  alpha: "Alpha",
  depth: "Depth",
  normal: "Normal",
  lambertian: "Lambertian",
  blinn_phong: "Blinn-Phong",
};
const minRenderDimension = 64;
const maxRenderDimension = 2048;

function setBusy(value) {
  busy = value;
  busyBadge.classList.toggle("hidden", !value);
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
  viewportReadout.textContent = [
    `${Number(camera.width || 0)} x ${Number(camera.height || 0)}`,
    `${Number(render.render_ms || 0).toFixed(1)} ms`,
    data.status || "ready",
  ].join("  /  ");
  statusBox.textContent = data.status || "ready";
  layerButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.view === data.view);
  });
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
    return;
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
  } catch (error) {
    statusBox.textContent = `error: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

function action(payload) {
  return postJson("/api/action", payload);
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
  resizeTimer = window.setTimeout(syncViewportSize, 180);
}

async function syncViewportSize() {
  const size = viewportRenderSize();
  if (size === null) {
    return;
  }
  if (busy) {
    scheduleViewportResize();
    return;
  }
  const camera = (snapshot && snapshot.camera) || {};
  if (Number(camera.width) === size.width && Number(camera.height) === size.height) {
    return;
  }
  await postJson("/api/resize", size);
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
    event.preventDefault();
    saveCurrent();
    return;
  }
  if (event.repeat || busy) {
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
  if (numberMap[key]) {
    event.preventDefault();
    setView(numberMap[key]);
    return;
  }
  const movementMap = {
    w: "forward",
    s: "backward",
    a: "left",
    d: "right",
    shift: "up",
  };
  if (movementMap[key]) {
    event.preventDefault();
    action({ action: "move", command: movementMap[key] });
    return;
  }
  if (key === "control") {
    pendingControlDown = true;
    event.preventDefault();
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
  if (event.key.toLowerCase() === "control" && pendingControlDown && !busy) {
    pendingControlDown = false;
    event.preventDefault();
    action({ action: "move", command: "down" });
  }
});

viewportWrap.addEventListener("pointerdown", (event) => {
  if (busy) {
    return;
  }
  let mode = null;
  if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
    mode = "pan";
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
  viewportWrap.setPointerCapture(event.pointerId);
});

viewportWrap.addEventListener("pointermove", (event) => {
  if (!drag || busy) {
    return;
  }
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  if (Math.abs(dx) < 1 && Math.abs(dy) < 1) {
    return;
  }
  drag.x = event.clientX;
  drag.y = event.clientY;
  action({ action: drag.mode, dx, dy });
});

viewportWrap.addEventListener("pointerup", () => {
  drag = null;
});

viewportWrap.addEventListener("pointerleave", () => {
  drag = null;
});

viewportWrap.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    if (busy) {
      return;
    }
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
