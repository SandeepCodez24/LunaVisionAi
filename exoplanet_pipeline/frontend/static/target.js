(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const { api, num, esc, Chart, gauge, bars, tag, countUp } = LV;

  const FEATS = [
    ["period", "Period (d)", 3], ["duration_hr", "Duration (h)", 2], ["depth_ppm", "Depth (ppm)", 0],
    ["SDE", "SDE", 1], ["SNR", "SNR", 1], ["transit_count", "Transits", 0], ["rp_rs", "Rp / R★", 4],
    ["odd_even_ratio", "Odd/even ratio", 2], ["secondary_ratio", "Secondary ratio", 2], ["centroid_proxy", "Centroid shift", 3],
    ["stellar_Teff", "Teff (K)", 0], ["stellar_rad", "Star radius (R☉)", 2], ["tess_mag", "TESS mag", 2],
  ];

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

    gauge($("#gauge"), d.p_transit); bars($("#probs"), d.probabilities);

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
    LV.reveal();
  }
  main();
})();
