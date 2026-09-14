"""
Staged, verbose smoke test of the real acquisition pipeline against live
MAST — run once, then delete. Runs each stage separately (instead of the
monolithic run_acquisition_pipeline) so failures/slowness are localized.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import acquisition as acq

def stage(name):
    print(f"\n===STAGE_START:{name}===", flush=True)
    return time.perf_counter()

def done(name, t0):
    print(f"===STAGE_DONE:{name} ({time.perf_counter()-t0:.1f}s)===", flush=True)

t0 = stage("parse_exofop_toi")
toi_df = acq.parse_exofop_toi()
done("parse_exofop_toi", t0)
print("toi_df rows:", len(toi_df))

t0 = stage("parse_nasa_confirmed")
nasa_df = acq.parse_nasa_confirmed()
done("parse_nasa_confirmed", t0)
print("nasa_df rows:", len(nasa_df))

# Pick 6 real TIC IDs from the TOI catalog that have NOT been downloaded yet,
# so this actually exercises the fresh-download path (not the resume-skip).
already = set()
lc_dir = acq.PROCESSED_DIR / "lc_raw"
if lc_dir.exists():
    already = {p.stem.replace("TIC_", "") for p in lc_dir.glob("TIC_*.npz")}
all_ids = list(toi_df["tic_id"].dropna().unique())
target_ids = [t for t in all_ids if t not in already][:6]
print("target_ids (fresh, not previously downloaded):", target_ids)

t0 = stage("crossmatch_tic")
tic_df = acq.crossmatch_tic(target_ids, batch_size=100)
done("crossmatch_tic", t0)
print("tic_df rows:", len(tic_df))

t0 = stage("download_light_curves")
manifest = acq.download_light_curves(
    tic_ids=target_ids,
    sector=1,
    output_dir=acq.PROCESSED_DIR / "lc_raw",
    concurrency=4,
)
done("download_light_curves", t0)
print("manifest entries:", len(manifest))

t0 = stage("build_unified_labels")
unified_df = acq.build_unified_labels(toi_df, nasa_df, None)
done("build_unified_labels", t0)
print("unified_df rows:", len(unified_df))

print("\n===ALL_STAGES_COMPLETE===", flush=True)
