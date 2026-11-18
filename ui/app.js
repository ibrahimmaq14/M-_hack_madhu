"use strict";

const el = (id) => document.getElementById(id);
const api = (new URLSearchParams(window.location.search).get("api") ||
  "http://127.0.0.1:8000").replace(/\/+$/, "");
const canvas = el("map");
const ctx = canvas.getContext("2d");
const statusLabel = el("status");
let data;
let drag;
let view = {scale: 1, x: 0, y: 0};

function status(state, text) {
  statusLabel.dataset.state = state;
  statusLabel.lastElementChild.textContent = text;
}

async function request(path, options) {
  const response = await fetch(api + path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === "string" ? payload.detail : "API returned HTTP " + response.status);
  }
  return response.json();
}

function showError(message) {
  el("error").hidden = false;
  el("error").textContent = message;
}

function projectGeometry(geometry) {
  const [lon0, lat0] = geometry.gps[0];
  const lonScale = Math.cos(lat0 * Math.PI / 180) * 111195;
  const project = ([lon, lat]) => [((lon - lon0 + 540) % 360 - 180) * lonScale, -(lat - lat0) * 111195];
  const gps = geometry.gps.map(project);
  const route = geometry.route.map((path) => path.map(project));
  const roads = geometry.roads.map((pair) => pair.map(project));
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  for (const [x, y] of gps.concat(route.flat())) {
    bounds[0] = Math.min(bounds[0], x);
    bounds[1] = Math.min(bounds[1], y);
    bounds[2] = Math.max(bounds[2], x);
    bounds[3] = Math.max(bounds[3], y);
  }
  return {gps, route, roads, bounds};
}

function fit() {
  if (!data) return;
  const [minX, minY, maxX, maxY] = data.bounds;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const scale = Math.min((w - 70) / Math.max(maxX - minX, 30), (h - 70) / Math.max(maxY - minY, 30));
  view = {scale, x: w / 2 - (minX + maxX) / 2 * scale, y: h / 2 - (minY + maxY) / 2 * scale};
  draw();
}

function toScreen([x, y]) {
  return [x * view.scale + view.x, y * view.scale + view.y];
}

function polyline(path) {
  ctx.moveTo(...toScreen(path[0]));
  path.slice(1).forEach((point) => ctx.lineTo(...toScreen(point)));
}

function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pixel = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * pixel);
  canvas.height = Math.round(h * pixel);
  ctx.scale(pixel, pixel);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.strokeStyle = "#20342b";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 48) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  for (let y = 0; y < h; y += 48) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();
  if (!data) return;
  if (el("roads").checked) {
    ctx.beginPath();
    ctx.strokeStyle = "#456256";
    ctx.lineWidth = 1;
    data.roads.forEach(polyline);
    ctx.stroke();
  }
  if (el("route").checked) {
    ctx.beginPath();
    ctx.strokeStyle = "#172c25";
    ctx.lineWidth = 7;
    data.route.forEach(polyline);
    ctx.stroke();
    ctx.strokeStyle = "#8ff0bd";
    ctx.lineWidth = 3.5;
    ctx.stroke();
  }
  if (el("gps").checked) {
    ctx.fillStyle = "#f0bd72";
    ctx.globalAlpha = .8;
    const stride = Math.max(1, Math.ceil(data.gps.length / 500));
    ctx.beginPath();
    for (let i = 0; i < data.gps.length; i += stride) {
      const [x, y] = toScreen(data.gps[i]);
      ctx.moveTo(x + 2, y);
      ctx.arc(x, y, 1.8, 0, 2 * Math.PI);
    }
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  const metres = 90 / view.scale;
  const power = 10 ** Math.floor(Math.log10(metres));
  const distance = Math.max(power, Math.floor(metres / power) * power);
  const scale = el("scale");
  scale.hidden = false;
  scale.style.width = Math.round(distance * view.scale) + "px";
  scale.textContent = distance >= 1000 ? distance / 1000 + " km" : distance + " m";
}

function zoom(factor, x = canvas.clientWidth / 2, y = canvas.clientHeight / 2) {
  if (!data) return;
  view.x = x - (x - view.x) * factor;
  view.y = y - (y - view.y) * factor;
  view.scale *= factor;
  draw();
}

["roads", "gps", "route"].forEach((layer) => el(layer).addEventListener("change", draw));
el("zoom-in").addEventListener("click", () => zoom(1.4));
el("zoom-out").addEventListener("click", () => zoom(1 / 1.4));
el("reset").addEventListener("click", fit);
canvas.addEventListener("wheel", (event) => {
  if (!data || !event.ctrlKey) return;
  event.preventDefault();
  const bounds = canvas.getBoundingClientRect();
  zoom(event.deltaY < 0 ? 1.2 : 1 / 1.2, event.clientX - bounds.left, event.clientY - bounds.top);
}, {passive: false});
canvas.addEventListener("pointerdown", (event) => {
  if (!data) return;
  drag = [event.clientX - view.x, event.clientY - view.y];
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!drag) return;
  view.x = event.clientX - drag[0];
  view.y = event.clientY - drag[1];
  draw();
});
canvas.addEventListener("lostpointercapture", () => { drag = undefined; });
new ResizeObserver(() => data ? fit() : draw()).observe(canvas);

function pct(value) {
  return value == null ? "—" : (100 * value).toFixed(1) + "%";
}

function render(result) {
  data = projectGeometry(result.geometry);
  el("placeholder").hidden = true;
  el("map-detail").textContent = result.decoder.matched_distance_km.toFixed(2) + " km of matched road";
  el("trip-tag").textContent = result.trip_id;
  el("accuracy").textContent = pct(result.metrics?.point_accuracy);
  el("similarity").textContent = pct(result.metrics?.sequence_similarity);
  el("continuity").textContent = pct(result.metrics?.route_continuity);
  el("observations").textContent = result.observations.toLocaleString();
  el("parts").textContent = result.decoder.parts.toLocaleString();
  el("elapsed").textContent = result.elapsed_seconds.toFixed(2) + " s";
  el("result-note").textContent = result.metrics
    ? "Gold marks are GPS observations. Green shows the reconstructed route."
    : "No server-side reference is available for this trip.";
  fit();
}

el("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  el("error").hidden = true;
  el("run").disabled = true;
  el("trip").disabled = true;
  status("busy", "Matching route");
  try {
    render(await request("/predict", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({trip_id: el("trip").value}),
    }));
    status("ready", "API connected");
  } catch (error) {
    showError(error.message);
    status("error", "Request failed");
  } finally {
    el("run").disabled = false;
    el("trip").disabled = false;
  }
});

async function init() {
  el("api-address").textContent = api;
  try {
    const [health, listing] = await Promise.all([request("/health"), request("/trips")]);
    el("trip").replaceChildren(...listing.trips.map((id) => new Option(id, id)));
    el("mode").textContent = listing.mode === "student_demo" ? "Student-test demo" : "Evaluator session";
    el("model").textContent = health.model + " · " + health.variant.toLowerCase() + " · Viterbi";
    el("trip").disabled = false;
    el("run").disabled = false;
    status("ready", "API connected");
  } catch (error) {
    showError("Could not connect to the PathSnap API. " + error.message);
    status("error", "API unavailable");
  }
}
init();
