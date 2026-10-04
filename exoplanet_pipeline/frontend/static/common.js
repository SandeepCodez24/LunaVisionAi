/* LunaVisionAI — shared helpers: api, starfield, count-up, reveal, charts, gauge. */
window.LV = (function () {
  "use strict";
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ease = (t) => 1 - Math.pow(1 - t, 3);

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { const j = await r.json(); msg = j.detail || msg; } catch (_) {}
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return r.json();
  }

  const num = (v, d = 2) => (v == null || !isFinite(v) ? "—" : Number(v).toFixed(d));
  const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ── Starfield with twinkle, mouse parallax and shooting stars ─────────── */
  function starfield() {
    const cv = document.getElementById("starfield");
    if (!cv) return;
    const ctx = cv.getContext("2d");
    let w, h, dpr, stars = [], shooters = [], mx = 0, my = 0, tx = 0, ty = 0;
    function resize() {
      dpr = Math.min(devicePixelRatio || 1, 2);
      w = cv.width = innerWidth * dpr; h = cv.height = innerHeight * dpr;
      const n = Math.round((innerWidth * innerHeight) / 5200);
      stars = Array.from({ length: n }, () => {
        const z = Math.random();
        return { x: Math.random() * w, y: Math.random() * h, z, r: (0.4 + z * 1.3) * dpr,
                 ph: Math.random() * 6.28, sp: 0.4 + Math.random() * 1.6,
                 hue: Math.random() < 0.12 ? "255,214,150" : Math.random() < 0.1 ? "170,190,255" : "243,239,228" };
      });
    }
    addEventListener("resize", resize); resize();
    addEventListener("pointermove", (e) => { tx = (e.clientX / innerWidth - 0.5); ty = (e.clientY / innerHeight - 0.5); });
    let last = 0;
    function frame(t) {
      mx += (tx - mx) * 0.04; my += (ty - my) * 0.04;
      ctx.clearRect(0, 0, w, h);
      for (const s of stars) {
        const a = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(t * 0.001 * s.sp + s.ph));
        const x = (s.x + mx * 40 * s.z * dpr + w) % w, y = (s.y + my * 40 * s.z * dpr + h) % h;
        ctx.fillStyle = `rgba(${s.hue},${(a * (0.25 + s.z * 0.75)).toFixed(3)})`;
        ctx.beginPath(); ctx.arc(x, y, s.r, 0, 6.283); ctx.fill();
      }
      if (!reduce && t - last > 4500 && Math.random() < 0.02) {
        last = t;
        shooters.push({ x: Math.random() * w * 0.8, y: Math.random() * h * 0.4, vx: (5 + Math.random() * 4) * dpr, vy: (2 + Math.random() * 2) * dpr, life: 1 });
      }
      shooters = shooters.filter((s) => s.life > 0);
      for (const s of shooters) {
        const g = ctx.createLinearGradient(s.x, s.y, s.x - s.vx * 14, s.y - s.vy * 14);
        g.addColorStop(0, `rgba(255,236,200,${s.life})`); g.addColorStop(1, "rgba(255,236,200,0)");
        ctx.strokeStyle = g; ctx.lineWidth = 1.6 * dpr; ctx.beginPath();
        ctx.moveTo(s.x, s.y); ctx.lineTo(s.x - s.vx * 14, s.y - s.vy * 14); ctx.stroke();
        s.x += s.vx; s.y += s.vy; s.life -= 0.018;
      }
      if (!reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    if (reduce) frame(0);
  }

  function countUp(el, to, { decimals = 0, duration = 1400, suffix = "" } = {}) {
    if (to == null || !isFinite(to)) { el.textContent = "—"; return; }
    if (reduce) { el.textContent = to.toFixed(decimals) + suffix; return; }
    const t0 = performance.now();
    (function tick(now) {
      const k = Math.min(1, (now - t0) / duration);
      el.textContent = (to * ease(k)).toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, decimals ? "" : ",") + suffix;
      if (k < 1) requestAnimationFrame(tick);
    })(t0);
  }

  function reveal(root = document) {
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); e.target.dispatchEvent(new Event("revealed")); io.unobserve(e.target); }
    }), { threshold: 0.12 });
    root.querySelectorAll(".reveal:not(.in)").forEach((el) => io.observe(el));
  }

  /* ── Canvas chart: scatter + line + markers, animated draw-in ─────────── */
  class Chart {
    constructor(canvas, opt = {}) {
      this.cv = canvas; this.ctx = canvas.getContext("2d");
      this.opt = Object.assign({ pad: { l: 52, r: 14, t: 12, b: 30 }, xLabel: "", yLabel: "", color: "#d99a35" }, opt);
      this.layers = []; this.progress = 1; this.scan = null;
      this._ro = new ResizeObserver(() => this.draw()); this._ro.observe(canvas);
    }
    setData({ layers, xDomain, yDomain }) {
      this.layers = layers;
      const all = layers.flatMap((l) => l.pts || []);
      const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
      const ext = (a) => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; };
      this.x = xDomain || ext(xs);
      if (yDomain) this.y = yDomain;
      else {
        const s = ys.slice().sort((a, b) => a - b);
        const lo = s[Math.floor(s.length * 0.002)] ?? s[0], hi = s[Math.floor(s.length * 0.998)] ?? s[s.length - 1];
        const m = (hi - lo) * 0.12 || 0.001; this.y = [lo - m, hi + m];
      }
      this.draw();
    }
    sx(v) { const { l, r } = this.opt.pad; return l + ((v - this.x[0]) / (this.x[1] - this.x[0] || 1)) * (this.W - l - r); }
    sy(v) { const { t, b } = this.opt.pad; return t + (1 - (v - this.y[0]) / (this.y[1] - this.y[0] || 1)) * (this.H - t - b); }
    draw() {
      const cv = this.cv, dpr = Math.min(devicePixelRatio || 1, 2);
      const cw = cv.clientWidth, ch = cv.clientHeight; if (!cw || !ch || !this.x) return;
      if (cv.width !== cw * dpr || cv.height !== ch * dpr) { cv.width = cw * dpr; cv.height = ch * dpr; }
      const c = this.ctx; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, cw, ch);
      this.W = cw; this.H = ch; const { l, r, t, b } = this.opt.pad;
      c.font = "10px 'IBM Plex Mono', monospace"; c.fillStyle = "#6e6c82"; c.strokeStyle = "rgba(243,239,228,.07)"; c.lineWidth = 1;
      // grid + ticks
      for (let i = 0; i <= 4; i++) {
        const y = t + ((ch - t - b) * i) / 4, v = this.y[1] - ((this.y[1] - this.y[0]) * i) / 4;
        c.beginPath(); c.moveTo(l, y); c.lineTo(cw - r, y); c.stroke();
        c.textAlign = "right"; c.fillText(v.toFixed(4), l - 6, y + 3);
      }
      for (let i = 0; i <= 5; i++) {
        const x = l + ((cw - l - r) * i) / 5, v = this.x[0] + ((this.x[1] - this.x[0]) * i) / 5;
        c.textAlign = "center"; c.fillText(v.toFixed(Math.abs(this.x[1] - this.x[0]) > 20 ? 0 : 1), x, ch - b + 16);
      }
      c.textAlign = "right"; c.fillText(this.opt.xLabel, cw - r, t + 10);
      c.save(); c.beginPath(); c.rect(l, t - 2, cw - l - r, ch - t - b + 4); c.clip();
      const reveal = this.progress;
      for (const L of this.layers) {
        if (L.type === "bands") {
          c.fillStyle = L.color || "rgba(217,154,53,.30)";
          for (const [a, bb] of L.bands) { const x0 = this.sx(a), x1 = this.sx(bb); c.fillRect(x0, t, Math.max(3, x1 - x0), ch - t - b); }
        } else if (L.type === "scatter") {
          c.fillStyle = L.color; const n = Math.floor(L.pts.length * reveal);
          for (let i = 0; i < n; i++) { const p = L.pts[i]; c.globalAlpha = L.alpha ?? 0.6; c.fillRect(this.sx(p[0]) - L.size / 2, this.sy(p[1]) - L.size / 2, L.size, L.size); }
          c.globalAlpha = 1;
        } else if (L.type === "line") {
          c.strokeStyle = L.color; c.lineWidth = L.width || 1.6; c.lineJoin = "round";
          if (L.glow) { c.shadowColor = L.color; c.shadowBlur = 10; }
          c.beginPath(); const n = Math.max(2, Math.floor(L.pts.length * reveal));
          for (let i = 0; i < n; i++) { const p = L.pts[i]; i ? c.lineTo(this.sx(p[0]), this.sy(p[1])) : c.moveTo(this.sx(p[0]), this.sy(p[1])); }
          c.stroke(); c.shadowBlur = 0;
        }
      }
      if (this.scan != null) {  // scanning highlight
        const x = this.sx(this.scan); const g = c.createLinearGradient(x - 40, 0, x, 0);
        g.addColorStop(0, "rgba(217,154,53,0)"); g.addColorStop(1, "rgba(217,154,53,.28)");
        c.fillStyle = g; c.fillRect(x - 40, t, 40, ch - t - b);
        c.fillStyle = "#ffd88a"; c.fillRect(x - 0.5, t, 1, ch - t - b);
      }
      c.restore();
    }
    animate(ms = 1600, after) {
      if (reduce) { this.progress = 1; this.draw(); after && after(); return; }
      const t0 = performance.now(); this.progress = 0;
      const tick = (now) => {
        const k = Math.min(1, (now - t0) / ms); this.progress = ease(k); this.draw();
        if (k < 1) requestAnimationFrame(tick); else after && after();
      };
      requestAnimationFrame(tick);
    }
    sweep(ms = 2200) {
      if (reduce) return;
      const t0 = performance.now(), [a, b] = this.x;
      const tick = (now) => {
        const k = (now - t0) / ms; if (k >= 1) { this.scan = null; this.draw(); return; }
        this.scan = a + (b - a) * ease(k); this.draw(); requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  }

  /* ── Radial probability gauge (SVG, injected) ─────────────────────────── */
  function gauge(el, value, label = "P(transit)") {
    const R = 76, C = 2 * Math.PI * R;
    el.innerHTML = `<svg width="180" height="180" viewBox="0 0 180 180"><defs><linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#8a6423"/><stop offset="1" stop-color="#ffd88a"/></linearGradient></defs>
      <circle class="track" cx="90" cy="90" r="${R}" stroke-width="10"/><circle class="arc" cx="90" cy="90" r="${R}" stroke-width="10" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg>
      <div class="val"><b>0%</b><span>${esc(label)}</span></div>`;
    const arc = el.querySelector(".arc"), b = el.querySelector("b");
    if (value == null) { b.textContent = "—"; return; }
    requestAnimationFrame(() => requestAnimationFrame(() => { arc.style.strokeDashoffset = C * (1 - value); }));
    countUp(b, value * 100, { decimals: 1, duration: 1400, suffix: "%" });
  }

  function bars(el, probs, order) {
    const entries = Object.entries(probs || {}).sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0));
    el.innerHTML = entries.map(([k, v]) => `<div class="bar"><span>${esc(k)}</span><div class="track"><div class="fillb" data-w="${((v ?? 0) * 100).toFixed(1)}"></div></div><span>${pct(v)}</span></div>`).join("");
    requestAnimationFrame(() => requestAnimationFrame(() => el.querySelectorAll(".fillb").forEach((f, i) => setTimeout(() => (f.style.width = f.dataset.w + "%"), i * 120))));
  }

  /* chart layer builders shared by both pages */
  function lightcurveLayers(lc, feat) {
    const pts = lc.time.map((t, i) => [t, lc.flux[i]]);
    const layers = [];
    const P = feat && feat.period, T0 = feat && feat.t0, dur = feat && feat.duration_hr ? feat.duration_hr / 24 : 0.1;
    if (P > 0 && T0 != null) {
      const bands = []; let n0 = Math.ceil((lc.t_min - T0) / P);
      for (let t = T0 + n0 * P; t <= lc.t_max; t += P) bands.push([t - dur / 2, t + dur / 2]);
      layers.push({ type: "bands", bands: bands.slice(0, 200) });
    }
    layers.push({ type: "scatter", pts, color: "rgba(168,166,189,.9)", size: 1.6, alpha: 0.5 });
    layers.push({ type: "line", pts, color: "#d99a35", width: 1.1, glow: false });
    return layers;
  }
  function foldLayers(fold) {
    return [
      { type: "scatter", pts: fold.hours.map((h, i) => [h, fold.flux[i]]), color: "rgba(168,166,189,.9)", size: 2, alpha: 0.45 },
      { type: "line", pts: fold.bin_hours.map((h, i) => [h, fold.bin_flux[i]]), color: "#ffd88a", width: 2.2, glow: true },
    ];
  }
  function foldDomain(fold) { return [-fold.half_hours, fold.half_hours]; }
  function foldYDomain(fold) {                       // frame the binned dip, not the scatter outliers
    const lo = Math.min(...fold.bin_flux), hi = Math.max(...fold.bin_flux), m = (hi - lo) * 0.9 || 0.0005;
    return [lo - m, hi + m * 0.8];
  }

  const LABEL_CLS = { Transit: "c0", Other: "c3", "Eclipsing binary": "c1", EB: "c1", Blend: "c2" };
  const tag = (name) => (name ? `<span class="tag ${LABEL_CLS[name] || ""}">${esc(name)}</span>` : `<span class="tag">—</span>`);

  const SPLIT = {
    "held-out":   { label: "held-out",   cls: "s-held",  note: "Held out from training, so this score is an honest estimate of how the model does on stars it hasn't seen." },
    "trained-on": { label: "trained on", cls: "s-train", note: "This star was in the training set, so the score is optimistic." },
    "unseen":     { label: "new star",   cls: "s-held",  note: "This star is in neither the training nor the test split, so the model has not seen it." },
    "unknown":    { label: "split unknown", cls: "", note: "This model was trained before train/held-out tracking existed, so we can't say whether it saw this star. Retrain to enable it." },
  };
  const splitTag = (sp) => { const m = SPLIT[sp]; return m ? `<span class="tag ${m.cls}" title="${esc(m.note)}">${esc(m.label)}</span>` : ""; };

  document.addEventListener("DOMContentLoaded", () => { starfield(); reveal(); });
  return { api, num, pct, esc, countUp, reveal, Chart, gauge, bars, lightcurveLayers, foldLayers, foldDomain, foldYDomain, tag, SPLIT, splitTag, reduce, ease };
})();
