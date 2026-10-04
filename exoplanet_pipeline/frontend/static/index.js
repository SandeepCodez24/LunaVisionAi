(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const { api, num, pct, esc, countUp, Chart, gauge, bars, tag } = LV;

  /* ── Hero: planet crosses the star, the light curve dips in sync ───────── */
  (function heroScene() {
    const cv = $("#heroScene"), ro = $("#sceneReadout");
    const ctx = cv.getContext("2d");
    const dpr = Math.min(devicePixelRatio || 1, 2);
    let W, H;
    function size() { W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr; }
    size(); addEventListener("resize", size);
    const CYCLE = 9000, SR = 54, PR = 11;                          // ms, star radius, planet radius (px)
    const trail = [];
    function flux(x, cx, cy) {                                      // limb-darkened occultation (approximate)
      const d = Math.abs(x - cx) / SR;
      if (d >= 1 + PR / SR) return 1;
      const cover = Math.max(0, Math.min(1, 1 - (d - PR / SR) / (2 * PR / SR)));
      return 1 - 0.018 * cover * (1 - 0.3 * d * d);
    }
    function frame(now) {
      const t = (now % CYCLE) / CYCLE;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const cx = W / 2, cy = H * 0.34;
      const px = -PR * 3 + (W + PR * 6) * t, py = cy + Math.sin(t * Math.PI * 2) * 6;
      // corona
      const g = ctx.createRadialGradient(cx, cy, SR * 0.6, cx, cy, SR * 2.6);
      g.addColorStop(0, "rgba(255,200,110,.55)"); g.addColorStop(1, "rgba(255,200,110,0)");
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, cy * 2 + 40);
      // star with granulation shimmer
      const sg = ctx.createRadialGradient(cx - 14, cy - 14, 4, cx, cy, SR);
      sg.addColorStop(0, "#fff4d0"); sg.addColorStop(.6, "#ffc468"); sg.addColorStop(1, "#d98a1c");
      ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(cx, cy, SR, 0, 7); ctx.fill();
      ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, SR, 0, 7); ctx.clip();
      for (let i = 0; i < 14; i++) {
        const a = i * 2.4 + now * 0.0003, r = (i % 5) * 9 + 6;
        ctx.fillStyle = `rgba(255,255,255,${0.05 + 0.04 * Math.sin(now * 0.002 + i)})`;
        ctx.beginPath(); ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a * 1.3) * r, 7, 0, 7); ctx.fill();
      }
      ctx.restore();
      // planet
      ctx.fillStyle = "#0a0b14"; ctx.beginPath(); ctx.arc(px, py, PR, 0, 7); ctx.fill();
      ctx.strokeStyle = "rgba(255,216,138,.25)"; ctx.lineWidth = 1; ctx.stroke();
      // light curve panel
      const top = H * 0.62, bh = H * 0.28, left = 18, right = W - 18;
      ctx.strokeStyle = "rgba(243,239,228,.08)"; ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(right, top); ctx.moveTo(left, top + bh); ctx.lineTo(right, top + bh); ctx.stroke();
      const f = flux(px, cx, cy), x = left + (right - left) * t;
      trail.push([x, top + bh * 0.1 + (1 - f) / 0.02 * bh * 0.8 + (Math.random() - 0.5) * 1.2]);
      if (trail.length > 1 && trail[trail.length - 1][0] < trail[trail.length - 2][0]) trail.length = 1;
      ctx.strokeStyle = "#d99a35"; ctx.lineWidth = 1.8; ctx.shadowColor = "#d99a35"; ctx.shadowBlur = 10; ctx.beginPath();
      trail.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke(); ctx.shadowBlur = 0;
      const last = trail[trail.length - 1];
      ctx.fillStyle = "#ffd88a"; ctx.beginPath(); ctx.arc(last[0], last[1], 3.2, 0, 7); ctx.fill();
      ro.textContent = "flux " + f.toFixed(4);
      if (!LV.reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  })();

  /* ── Overview + model panel ────────────────────────────────────────────── */
  async function loadOverview() {
    let o;
    try { o = await api("/api/overview"); } catch (e) { $("#sTotal").textContent = "offline"; return; }
    const t = o.model && o.model.test;
    const io = new IntersectionObserver((es) => es.forEach((en) => {
      if (!en.isIntersecting) return; io.disconnect();
      countUp($("#sTotal"), o.total); countUp($("#sReal"), o.real); countUp($("#sCand"), o.candidates);
      countUp($("#sAuc"), t && t.roc_auc, { decimals: 3 });
    }), { threshold: 0.3 });
    io.observe($(".stats"));
    if (o.scoring_error) $("#count").textContent = "Classifier unavailable: " + o.scoring_error;
    if (!t) return;

    $("#metrics").innerHTML = [["ROC-AUC", t.roc_auc, 3], ["PR-AUC", t.pr_auc, 3], ["Brier score", t.brier, 3], ["Held-out stars", t.n, 0]]
      .map(([k, v, d]) => `<div class="card metric"><b data-v="${v}" data-d="${d}">0</b><span>${k}</span></div>`).join("");
    $("#metrics").addEventListener("revealed", () => $("#metrics").querySelectorAll("b").forEach((b) => countUp(b, +b.dataset.v, { decimals: +b.dataset.d })), { once: true });

    const m = t.confusion_matrix, labs = Object.keys(t.class_counts).map((k) => ({ 0: "Transit", 1: "EB", 2: "Blend", 3: "Other" }[k] || k));
    const max = Math.max(...m.flat());
    $("#cm").innerHTML = `<span></span><span>pred ${labs[0]}</span><span>pred ${labs[1]}</span>` +
      m.map((row, i) => `<span>true ${labs[i]}</span>` + row.map((v, j) => `<div class="cell" style="--a:${(0.08 + 0.7 * v / max).toFixed(2)};transition-delay:${(i * 2 + j) * 120}ms">${v}</div>`).join("")).join("");
    $("#cm").closest(".card").addEventListener("revealed", () => $("#cm").classList.add("in"), { once: true });

    const tf = (o.model.top_features || []).slice(0, 8), top = Math.max(...tf.map((f) => f.shap || f.permutation || 0)) || 1;
    $("#feats").innerHTML = tf.map((f) => { const v = f.shap ?? f.permutation ?? 0;
      return `<div class="feat-row"><span>${esc(f.feature)}</span><div class="track" style="height:8px;border-radius:99px;background:var(--surface-2);overflow:hidden"><div class="fillb" data-w="${(v / top * 100).toFixed(1)}" style="height:100%;width:0;background:var(--accent);border-radius:99px;transition:width 1.2s var(--ease)"></div></div><span>${v.toFixed(3)}</span></div>`; }).join("");
    $("#feats").closest(".card").addEventListener("revealed", () => setTimeout(() => $("#feats").querySelectorAll(".fillb").forEach((f, i) => setTimeout(() => (f.style.width = f.dataset.w + "%"), i * 90)), 150), { once: true });
  }

  /* ── Candidate explorer ────────────────────────────────────────────────── */
  const state = { q: "", label: "", min: 0, sort: "p_transit", order: "desc", offset: 0, limit: 25, total: 0 };
  let reqId = 0, debounce;
  function rowHtml(r, i) {
    const p = r.p_transit;
    return `<tr data-tic="${esc(r.tic_id)}" style="animation-delay:${i * 28}ms" tabindex="0">
      <td class="tic">${esc(r.tic_id.replace("TIC_", ""))}</td><td>${tag(r.label_name)}</td>
      <td><div class="pbar"><div class="track"><div class="fillb" style="--w:${p == null ? 0 : (p * 100).toFixed(1)}%"></div></div><span class="mono">${pct(p)}</span></div></td>
      <td>${num(r.period, 3)}</td><td>${num(r.depth_ppm, 0)}</td><td>${num(r.SDE, 1)}</td></tr>`;
  }
  async function loadRows(append) {
    const id = ++reqId;
    if (!append) { state.offset = 0; $("#rows").innerHTML = '<tr class="skeleton"><td colspan="6"></td></tr>'.repeat(5); }
    const qs = new URLSearchParams({ q: state.q, min_p: state.min, sort: state.sort, order: state.order, limit: state.limit, offset: state.offset });
    if (state.label !== "") qs.set("label", state.label);
    try {
      const d = await api("/api/targets?" + qs);
      if (id !== reqId) return;
      state.total = d.total;
      const html = d.items.map((r, i) => rowHtml(r, i)).join("");
      if (append) $("#rows").insertAdjacentHTML("beforeend", html); else $("#rows").innerHTML = html || '<tr><td colspan="6" class="muted" style="text-align:center;padding:32px">No matching stars.</td></tr>';
      const shown = $("#rows").querySelectorAll("tr[data-tic]").length;
      $("#count").textContent = `${shown.toLocaleString()} of ${d.total.toLocaleString()} stars`;
      $("#more").hidden = shown >= d.total;
    } catch (e) { if (id === reqId) $("#rows").innerHTML = `<tr><td colspan="6"><div class="error-box">${esc(e.message)}</div></td></tr>`; }
  }
  $("#rows").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-tic]"); if (tr) location.href = "/target?tic=" + tr.dataset.tic; });
  $("#rows").addEventListener("keydown", (e) => { if (e.key === "Enter") { const tr = e.target.closest("tr[data-tic]"); if (tr) location.href = "/target?tic=" + tr.dataset.tic; } });
  $("#q").addEventListener("input", (e) => { clearTimeout(debounce); debounce = setTimeout(() => { state.q = e.target.value.trim(); loadRows(); }, 250); });
  $("#fLabel").addEventListener("change", (e) => { state.label = e.target.value; loadRows(); });
  $("#fMin").addEventListener("input", (e) => { state.min = +e.target.value; $("#fMinVal").textContent = Math.round(state.min * 100) + "%"; clearTimeout(debounce); debounce = setTimeout(() => loadRows(), 200); });
  $("#more").addEventListener("click", () => { state.offset += state.limit; loadRows(true); });
  document.querySelectorAll("th[data-sort]").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.sort; state.order = state.sort === k && state.order === "desc" ? "asc" : "desc"; state.sort = k;
    document.querySelectorAll("th").forEach((x) => x.classList.toggle("sorted", x === th)); loadRows();
  }));

  /* ── Analyze console ───────────────────────────────────────────────────── */
  const stepper = $("#stepper");
  const STAGE_ICON = { done: "✓", error: "!", skip: "–" };
  function renderSteps(stages) {
    if (!stepper.querySelector(".step")) {
      stepper.insertAdjacentHTML("beforeend", stages.map((s) => `<li class="step" data-id="${s.id}"><span class="dot"></span>${esc(s.label)}<span class="info"></span></li>`).join(""));
    }
    let lastDone = -1;
    stages.forEach((s, i) => {
      const li = stepper.querySelector(`[data-id="${s.id}"]`);
      li.className = "step " + (s.state === "start" ? "running" : s.state === "pending" ? "" : s.state);
      li.querySelector(".dot").textContent = STAGE_ICON[s.state] || "";
      const inf = s.info || {}; let txt = inf.reason || "";
      if (s.id === "acquire" && s.state === "done") txt = { mast: "downloaded from MAST", upload: "your uploaded file", disk: "already on server", detrended: "already on server" }[inf.source] || "";
      if (s.id === "detect_features" && s.state === "done") txt = `period ${num(inf.period, 3)} d · SDE ${num(inf.SDE, 1)}`;
      if (s.id === "classify" && s.state === "done") txt = `P(transit) ${pct(inf.p_transit)}`;
      li.querySelector(".info").textContent = txt;
      if (["done", "skip"].includes(s.state)) lastDone = i;
    });
    const items = stepper.querySelectorAll(".step"), running = stages.findIndex((s) => s.state === "start");
    const upto = running >= 0 ? running : lastDone;
    $("#stepFill").style.height = upto < 0 ? 0 : (items[upto].offsetTop + 14 - 12) + "px";
  }
  function showResult(res) {
    const el = $("#result");
    el.innerHTML = `<div class="result-grid"><div class="gauge" id="g"></div><div>
      <span class="verdict ${res.is_candidate ? "cand" : "non"}">${res.is_candidate ? "Transit candidate" : "Not a candidate"}</span>
      <h3 style="font-size:26px;margin:10px 0 2px">${esc(res.tic_id.replace("_", " "))}</h3>
      <div class="muted mono" style="font-size:12px">P = ${num(res.features.period, 3)} d · depth ${num(res.features.depth_ppm, 0)} ppm · SDE ${num(res.features.SDE, 1)} · ${res.features.transit_count ?? "—"} transits</div>
      <div class="bars" id="pb"></div></div></div>
      <div class="mini-charts"><canvas id="cLc"></canvas><canvas id="cFold"></canvas></div>
      <div style="margin-top:16px"><a class="btn" href="/target?tic=${esc(res.tic_id)}">Open full analysis →</a></div>`;
    gauge($("#g"), res.p_transit); bars($("#pb"), res.probabilities);
    const lc = new Chart($("#cLc"), { pad: { l: 46, r: 8, t: 8, b: 22 }, xLabel: "time (BTJD)" });
    lc.setData({ layers: LV.lightcurveLayers(res.lightcurve, res.features) }); lc.animate(1500);
    if (res.fold) {
      const fc = new Chart($("#cFold"), { pad: { l: 46, r: 8, t: 8, b: 22 }, xLabel: "hours from mid-transit" });
      fc.setData({ layers: LV.foldLayers(res.fold), xDomain: LV.foldDomain(res.fold), yDomain: LV.foldYDomain(res.fold) });
      fc.animate(1400, () => fc.sweep(1800));
    }
  }
  let es = null;
  function startJob(submit, note) {
    if (es) es.close();
    stepper.querySelectorAll(".step").forEach((n) => n.remove()); $("#stepFill").style.height = 0;
    $("#result").innerHTML = `<div class="placeholder"><span class="mono">Running pipeline…</span><span>${esc(note || "The transit search takes ~20–30 s per star.")}</span></div>`;
    $("#runBtn").disabled = true; $("#drop").classList.add("busy");
    const idle = () => { $("#runBtn").disabled = false; $("#drop").classList.remove("busy"); };
    submit()
      .then(({ id }) => {
        es = new EventSource(`/api/jobs/${id}/events`);
        es.onmessage = (m) => {
          const j = JSON.parse(m.data); renderSteps(j.stages);
          if (j.status === "done") { es.close(); idle(); showResult(j.result); loadRecent(); }
          if (j.status === "failed") { es.close(); idle(); $("#result").innerHTML = `<div class="error-box">${esc(j.error)}</div>`; }
        };
        es.onerror = () => { es.close(); idle(); };
      })
      .catch((e) => { idle(); $("#result").innerHTML = `<div class="error-box">${esc(e.message)}</div>`; });
  }
  function runJob(tic) {
    startJob(() => api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tic_id: tic }) }),
             "Stars not on the server are fetched from MAST first, which can add a minute or more.");
  }
  function runUpload(file) {
    if (!file) return;
    const fd = new FormData(); fd.append("file", file);
    startJob(() => api("/api/jobs/upload", { method: "POST", body: fd }), `Checking ${file.name}, then running the pipeline.`);
  }

  // tabs
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    ["paneTic", "paneUpload"].forEach((id) => ($("#" + id).hidden = id !== t.dataset.pane));
  }));
  // drop zone
  const drop = $("#drop"), fileInput = $("#fileInput");
  fileInput.addEventListener("change", () => { runUpload(fileInput.files[0]); fileInput.value = ""; });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => runUpload(e.dataTransfer.files[0]));

  async function loadRecent() {
    try {
      const d = await api("/api/analyses?limit=8");
      $("#recentWrap").hidden = !d.items.length;
      $("#recent").innerHTML = d.items.map((a) => `<a class="chip" href="/target?tic=${esc(a.tic_id)}" title="${esc(a.pred || "")}">${esc(a.tic_id.replace("TIC_", ""))} · ${pct(a.p_transit)}</a>`).join("");
    } catch (_) {}
  }
  $("#jobForm").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#ticInput").value.trim(); if (v) runJob(v); });

  async function loadSamples() {
    try {
      const d = await api("/api/targets?limit=5&min_p=0.9&sort=SDE");
      $("#samples").innerHTML = d.items.map((r) => `<button class="chip" type="button" data-tic="${esc(r.tic_id)}">${esc(r.tic_id.replace("TIC_", ""))}</button>`).join("");
      $("#samples").addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (!c) return; $("#ticInput").value = c.dataset.tic.replace("TIC_", ""); document.querySelectorAll(".chip").forEach((x) => x.classList.toggle("active", x === c)); runJob(c.dataset.tic); });
    } catch (_) {}
  }

  loadOverview(); loadRows(); loadSamples(); loadRecent();
})();
