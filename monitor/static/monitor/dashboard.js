(() => {
  const POLL_MS = 3000;
  const detail = document.getElementById("detail");
  const placeholder = document.getElementById("placeholder");
  const tpl = document.getElementById("detail-tpl");
  const addDialog = document.getElementById("add-dialog");
  const deviceButtons = () => Array.from(document.querySelectorAll(".device"));

  let selectedId = null;
  let liveTimer = null;
  let historyCursor = null;
  let root = null;

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const bind = (name) => root.querySelector(`[data-bind="${name}"]`);
  const label = (type) => ({ fire: "Fire", smoke: "Smoke", none: "Clear" }[type] || type);
  const pct = (c) => (c === null || c === undefined ? "n/a" : `${(c * 100).toFixed(1)}%`);
  const when = (iso) => new Date(iso).toLocaleString();

  async function getJSON(url) {
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  }

  /* ---------- add device dialog ---------- */
  document.getElementById("open-add").addEventListener("click", () => addDialog.showModal());
  document.getElementById("cancel-add").addEventListener("click", () => addDialog.close());
  if (addDialog.hasAttribute("data-open-on-load")) addDialog.showModal();

  /* ---------- device selection ---------- */
  function select(id) {
    selectedId = String(id);
    deviceButtons().forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === selectedId)));
    history.replaceState(null, "", `?device=${selectedId}`);

    clearInterval(liveTimer);
    detail.replaceChildren(tpl.content.cloneNode(true));
    root = detail;

    bind("rules-link").href = `/devices/${selectedId}/rules/`;
    bind("delete-form").action = `/devices/${selectedId}/delete/`;
    bind("detections-only").addEventListener("change", () => loadHistory(true));
    bind("more").addEventListener("click", () => loadHistory(false));

    refreshLive();
    loadHistory(true);
    liveTimer = setInterval(refreshLive, POLL_MS);
  }

  deviceButtons().forEach((b) => b.addEventListener("click", () => select(b.dataset.id)));

  /* ---------- live card ---------- */
  function describe(data) {
    const { latest, verification, device } = data;
    if (!data.model_ready) return { tone: "idle", state: "Model not loaded", sub: "Place model.pt in the models folder to start predicting." };
    if (!latest) return { tone: "idle", state: "Waiting for the first snapshot", sub: `A snapshot is taken every ${device.snapshot_interval} seconds.` };
    if (device.status === "inactive") return { tone: "idle", state: "Camera unreachable", sub: "The last attempt to read the RTSP stream failed. Retrying on the next cycle." };
    if (latest.alarm_raised) return { tone: "alarm", state: `${label(latest.alarm_type)} alarm raised`, sub: `Confirmed at ${pct(latest.confidence)} confidence.` };
    if (latest.alarm_type !== "none") {
      const counting = verification ? `Verifying ${verification.count} of ${verification.required} snapshots.` : "Below the alarm threshold, so it does not count yet.";
      return { tone: "verify", state: `${label(latest.alarm_type)} detected`, sub: `${pct(latest.confidence)} confidence. ${counting}` };
    }
    return { tone: "clear", state: "Clear", sub: "No fire or smoke in the latest snapshot." };
  }

  function renderMeter(data) {
    const meter = bind("meter");
    const v = data.verification;
    const latest = data.latest;
    if (!v && !(latest && latest.alarm_raised)) { meter.hidden = true; return; }

    const required = v ? v.required : data.rules[latest.alarm_type].temporal_frames;
    const filled = v ? v.count : required;
    const done = !v;
    const segs = bind("segments");
    segs.replaceChildren();
    for (let i = 0; i < required; i++) segs.appendChild(el("span", `segment${i < filled ? (done ? " done" : " on") : ""}`));
    const type = v ? v.type : latest.alarm_type;
    bind("meter-label").textContent = done
      ? `${label(type)} confirmed after ${required} consecutive snapshots.`
      : `${label(type)} seen in ${filled} of ${required} consecutive snapshots needed for an alarm.`;
    meter.hidden = false;
  }

  function renderRules(rules) {
    const body = bind("rules");
    body.replaceChildren();
    ["fire", "smoke"].forEach((key) => {
      const r = rules[key];
      if (!r) return;
      const tr = el("tr", r.is_enabled ? "" : "off");
      tr.append(el("td", "", `${label(key)}${r.is_enabled ? "" : " (off)"}`), el("td", "", r.threshold.toFixed(2)), el("td", "", String(r.temporal_frames)));
      body.appendChild(tr);
    });
  }

  function renderRecent(items) {
    const list = bind("recent");
    list.replaceChildren();
    if (!items.length) { list.appendChild(el("li", "empty", "No predictions yet.")); return; }
    items.forEach((p) => {
      const cls = p.alarm_raised ? "chip chip-alarm" : p.alarm_type !== "none" ? "chip chip-detected" : "chip";
      const li = el("li", cls);
      li.append(el("span", "", p.alarm_raised ? `${label(p.alarm_type)} alarm` : label(p.alarm_type)), el("small", "", p.alarm_type === "none" ? when(p.created_at) : `${pct(p.confidence)} at ${new Date(p.created_at).toLocaleTimeString()}`));
      list.appendChild(li);
    });
  }

  async function refreshLive() {
    const id = selectedId;
    try {
      const data = await getJSON(`/api/devices/${id}/live/`);
      if (id !== selectedId) return;
      bind("name").textContent = data.device.name;
      bind("url").textContent = data.device.rtsp_url;

      const view = describe(data);
      bind("live").dataset.tone = view.tone;
      bind("state").textContent = view.state;
      bind("state-sub").textContent = view.sub;

      const img = bind("image");
      const empty = bind("image-empty");
      if (data.latest && data.latest.snapshot_url) {
        if (img.getAttribute("src") !== data.latest.snapshot_url) img.src = data.latest.snapshot_url;
        img.hidden = false; empty.hidden = true;
      }

      renderMeter(data);
      renderRules(data.rules);
      renderRecent(data.recent);
      updateListStatus(data.device);
    } catch (err) {
      bind("state-sub").textContent = "Could not reach the server. Retrying.";
    }
  }

  function updateListStatus(device) {
    const btn = deviceButtons().find((b) => b.dataset.id === String(device.id));
    if (btn) btn.querySelector(".dot").className = `dot dot-${device.status}`;
  }

  /* ---------- history ---------- */
  async function loadHistory(reset) {
    const id = selectedId;
    const body = bind("history");
    const detectionsOnly = bind("detections-only").checked ? 1 : 0;
    if (reset) { body.replaceChildren(); historyCursor = null; }
    const before = historyCursor ? `&before=${historyCursor}` : "";
    try {
      const data = await getJSON(`/api/devices/${id}/predictions/?detections_only=${detectionsOnly}&limit=20${before}`);
      if (id !== selectedId) return;
      data.predictions.forEach((p) => body.appendChild(historyRow(p)));
      historyCursor = data.next_before;
      bind("more").hidden = !data.next_before;
      bind("history-empty").hidden = body.children.length > 0;
    } catch (err) {
      bind("history-empty").hidden = false;
      bind("history-empty").textContent = "Could not load history.";
    }
  }

  function historyRow(p) {
    const tr = el("tr");
    const result = el("td");
    result.appendChild(el("span", `tag tag-${p.alarm_type}`, label(p.alarm_type)));
    const alarm = el("td");
    alarm.appendChild(p.alarm_raised ? el("span", "tag tag-alarm", "Raised") : el("span", "muted", "No"));
    const snap = el("td");
    if (p.snapshot_url) {
      const a = el("a");
      a.href = p.snapshot_url; a.target = "_blank"; a.rel = "noopener";
      const img = el("img"); img.src = p.snapshot_url; img.alt = `Snapshot from ${when(p.created_at)}`; img.loading = "lazy";
      a.appendChild(img); snap.appendChild(a);
    } else snap.textContent = "None";
    tr.append(el("td", "", when(p.created_at)), result, el("td", "", pct(p.confidence)), alarm, snap);
    return tr;
  }

  /* ---------- keep the device list statuses fresh ---------- */
  setInterval(async () => {
    try {
      const data = await getJSON("/api/devices/");
      data.devices.forEach(updateListStatus);
    } catch (err) { /* ignore transient errors */ }
  }, 10000);

  const initial = detail.dataset.selected;
  if (initial && deviceButtons().some((b) => b.dataset.id === initial)) select(initial);
})();
