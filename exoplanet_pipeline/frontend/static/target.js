(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const { api, num, esc, Chart, gauge, bars, tag, countUp, splitTag, SPLIT, renderStepper } = LV;

  const FEATS = [
    ["period", "Period (d)", 3], ["duration_hr", "Duration (h)", 2], ["depth_ppm", "Depth (ppm)", 0],
    ["SDE", "SDE", 1], ["SNR", "SNR", 1], ["transit_count", "Transits", 0], ["rp_rs", "Rp / R★", 4],
    ["odd_even_ratio", "Odd/even ratio", 2], ["secondary_ratio", "Secondary ratio", 2], ["centroid_proxy", "Centroid shift", 3],
    ["stellar_Teff", "Teff (K)", 0], ["stellar_rad", "Star radius (R☉)", 2], ["tess_mag", "TESS mag", 2],
  ];

  const PLOT_TITLES = { "1_lightcurve": "Light curve", "2_periodogram": "TLS periodogram", "3_fold": "Phase fold + model", "4_corner": "MCMC corner",
    "5_odd_even": "Odd / even depth", "6_secondary": "Secondary eclipse", "7_centroid": "Centroid motion" };
  const FIT_ROWS = [["rp_rearth", "Planet radius (R⊕)", 2], ["a_au", "Semi-major axis (AU)", 4], ["teq_k", "Equilibrium temp (K)", 0],
    ["inc_deg", "Inclination (°)", 2], ["b", "Impact parameter", 2], ["t14_hr", "Duration T14 (h)", 2], ["depth_ppm", "Depth (ppm)", 0], ["snr", "Transit SNR", 1]];
  const ICON = { PASS: "✓", FAIL: "✗", "N/A": "–" };
  let current = null;

  function renderVetting(d) {
    const v = d.vetting, body = $("#vetBody"), pdf = $("#pdfBtn");
    if (!v) { body.innerHTML = ""; $("#vetIntro").hidden = false; return; }
    $("#vetIntro").hidden = true; $("#genBtn").textContent = "Re-run report";
    if (d.has_report) { pdf.hidden = false; pdf.href = `/api/targets/${d.tic_id}/report`; }
    const cls = v.verdict === "PASS" ? "PASS" : v.verdict === "CAUTION" ? "CAUTION" : "FP";
    const q = `?v=${Math.round(v.made || 0)}`;
    body.innerHTML = `<div style="margin-top:14px"><span class="vbadge ${cls}">${esc(v.verdict)}</span>
        <span class="muted mono" style="font-size:12px;margin-left:12px">${esc(v.method || "")}${v.converged ? " · converged" : ""}${v.size_class ? " · " + esc(v.size_class) : ""}</span></div>
      <div class="checks">${v.checks.map(([n, st, det], i) => { const k = st === "N/A" ? "NA" : st;
        return `<div class="check" style="animation-delay:${i * 90}ms"><span class="ic ${k}">${ICON[st] || ""}</span><b>${esc(n)}</b><span>${esc(det)}</span></div>`; }).join("")}</div>
      <div class="fit-table">${FIT_ROWS.filter(([k]) => v.fit[k]).map(([k, label, dec]) => { const f = v.fit[k];
        return `<div class="feat"><b>${num(f.med, dec)}</b><span>${label}</span><small>+${num(f.hi, dec)} / −${num(f.lo, dec)}</small></div>`; }).join("")}</div>
      <div class="gallery">${(d.plots || []).map((n, i) => `<figure class="shot" style="animation-delay:${i * 110}ms;margin:0" data-src="/api/targets/${esc(d.tic_id)}/plots/${esc(n)}${q}">
        <img loading="lazy" alt="${esc(PLOT_TITLES[n] || n)}" src="/api/targets/${esc(d.tic_id)}/plots/${esc(n)}${q}"><span>${esc(PLOT_TITLES[n] || n)}</span></figure>`).join("")}</div>`;
    body.querySelectorAll(".shot").forEach((el) => el.addEventListener("click", () => {
      const lb = document.createElement("div"); lb.className = "lightbox";
      lb.innerHTML = `<button class="btn" type="button">Close ✕</button><img alt="" src="${el.dataset.src}">`;
      const close = () => { lb.remove(); document.removeEventListener("keydown", onKey); }, onKey = (e) => e.key === "Escape" && close();
      lb.addEventListener("click", close); document.addEventListener("keydown", onKey); document.body.appendChild(lb);
    }));
  }

  async function refresh() { current = await api("/api/targets/" + encodeURIComponent(current.tic_id)); renderVetting(current); }

  function watchReport(jobId) {
    const ol = $("#rstepper"), btn = $("#genBtn"); ol.hidden = false; btn.disabled = true;
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    const stop = () => { es.close(); btn.disabled = false; };
    es.onmessage = async (m) => {
      const j = JSON.parse(m.data);
      renderStepper(ol, $("#rfill"), j.stages, (s) => s.state === "done" && s.id === "fit" ? `${s.info.method || ""}${s.info.converged ? ", converged" : ""}` : s.state === "done" && s.id === "report" ? s.info.verdict : (s.info && s.info.reason) || "");
      if (j.status === "done") { stop(); await refresh(); setTimeout(() => { ol.hidden = true; ol.querySelectorAll(".step").forEach((n) => n.remove()); $("#rfill").style.height = 0; }, 900); }
      if (j.status === "failed") { stop(); $("#vetBody").innerHTML = `<div class="error-box" style="margin-top:14px">${esc(j.error)}</div>`; }
    };
    es.onerror = stop;
  }

  async function startReport(force) {
    const btn = $("#genBtn"); btn.disabled = true; $("#vetBody").innerHTML = "";
    try {
      const r = await api(`/api/targets/${encodeURIComponent(current.tic_id)}/report${force ? "?force=true" : ""}`, { method: "POST" });
      if (r.ready) { btn.disabled = false; await refresh(); } else watchReport(r.id);
    } catch (e) { btn.disabled = false; $("#vetBody").innerHTML = `<div class="error-box" style="margin-top:14px">${esc(e.message)}</div>`; }
  }

  async function main() {
    const tic = new URLSearchParams(location.search).get("tic");
    if (!tic) { $("#state").textContent = "No target given. Pick one from the candidate list."; return; }
    let d;
    try { d = await api("/api/targets/" + encodeURIComponent(tic)); }
    catch (e) { $("#state").innerHTML = `<div class="error-box">${LV.esc(e.message)}</div>`; return; }

    document.title = d.tic_id.replace("_", " ") + " · LunaVisionAI";
    $("#state").hidden = true; $("#content").hidden = false;
    $("#kind").textContent = ({ synthetic: "Synthetic injection", upload: "Uploaded light curve" }[d.kind] || "TESS target") + (d.sector > 0 ? ` · Sector ${d.sector}` : "");
    $("#title").textContent = d.tic_id.replace("_", " ");
    $("#tags").innerHTML = `${tag(d.label_name) .replace('class="tag', 'title="Catalog label" class="tag')}${d.pred ? `<span class="tag" title="Classifier prediction">model: ${esc(d.pred)}</span>` : ""}`;
    if (d.has_report) { const a = $("#pdf"); a.hidden = false; a.href = `/api/targets/${d.tic_id}/report`; }

    const sn = $("#splitNote"), m = SPLIT[d.split];
    if (d.kind !== "synthetic" && (m || d.detected === false)) {
      sn.hidden = false; sn.classList.toggle("warn", d.split === "trained-on");
      sn.innerHTML = (d.detected === false ? "<b>No transit detected</b>: the search found no significant period, so the model gives no score for this star. " : "") +
        (m && d.detected !== false ? `<b>${esc(m.label)}.</b> ${esc(m.note)} ` : "") +
        (d.detected === false ? "" : `<span class="muted">The score is how much the features resemble catalog planet-side stars, not the probability of a planet.</span>`);
    }
    if (d.split) $("#tags").insertAdjacentHTML("beforeend", splitTag(d.split));
    gauge($("#gauge"), d.p_transit, d.detected === false ? "no detection" : "model score"); bars($("#probs"), d.probabilities);

    $("#feats").innerHTML = FEATS.map(([k, label, dec]) => {
      const v = d.features[k]; return `<div class="feat"><b data-k="${k}" data-v="${v ?? ""}" data-d="${dec}">${v == null ? "—" : "0"}</b><span>${label}</span></div>`;
    }).join("");
    $("#feats").closest(".card").addEventListener("revealed", () => $("#feats").querySelectorAll("b").forEach((b) => {
      if (b.dataset.v !== "") countUp(b, +b.dataset.v, { decimals: +b.dataset.d, duration: 1200 });
    }), { once: true });

    const lc = new Chart($("#cLc"), { xLabel: "time (BTJD, days)" });
    lc.setData({ layers: LV.lightcurveLayers(d.lightcurve, d.features) });
    $("#cLc").closest(".card").addEventListener("revealed", () => lc.animate(2200), { once: true });

    if (d.fold) {
      const fc = new Chart($("#cFold"), { xLabel: "hours from mid-transit" });
      fc.setData({ layers: LV.foldLayers(d.fold), xDomain: LV.foldDomain(d.fold), yDomain: LV.foldYDomain(d.fold) });
      $("#cFold").closest(".card").addEventListener("revealed", () => fc.animate(1800, () => fc.sweep(2600)), { once: true });
      $("#foldNote").textContent = `Folded on P = ${num(d.features.period, 4)} d, T0 = ${num(d.features.t0, 3)}.`;
    } else {
      $("#cFold").style.display = "none";
      $("#foldNote").textContent = "No transit period was found for this star, so there is nothing to fold.";
    }
    current = d;
    const noFit = d.detected === false || !d.fold;
    if (noFit) { $("#genBtn").disabled = true; $("#vetIntro").textContent = "No transit period was found for this star, so there is nothing to fit."; }
    $("#genBtn").addEventListener("click", () => startReport(!!current.vetting));
    renderVetting(d);
    LV.reveal();
  }
  main();
})();
