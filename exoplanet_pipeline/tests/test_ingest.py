"""Upload parsing/validation. Run: .venv/bin/python -m pytest tests -q"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ingest  # noqa: E402


def _csv(n=500, header="time,flux", span=20.0, flux=None):
    t = np.linspace(0, span, n)
    f = 1 + 0.001 * np.sin(t * 3) if flux is None else flux
    rows = "\n".join(f"{a:.5f},{b:.6f}" for a, b in zip(t, f))
    return (header + "\n" + rows if header else rows).encode()


def test_csv_with_header_parses():
    lc = ingest.parse_light_curve("a.csv", _csv())
    assert len(lc["time"]) == 500 and np.all(np.diff(lc["time"]) > 0)


def test_headerless_two_columns_parse():
    assert len(ingest.parse_light_curve("a.txt", _csv(header=""))["time"]) == 500


def test_alternate_column_names_and_julian_date_to_btjd():
    t = 2458000.0 + np.linspace(0, 20, 400)
    data = ("BJD,PDCSAP_FLUX\n" + "\n".join(f"{a:.5f},{1 + 1e-3 * (i % 9):.5f}" for i, a in enumerate(t))).encode()
    lc = ingest.parse_light_curve("a.csv", data)
    assert 1000 <= lc["time"][0] < 1100                    # BJD 2458000 -> BTJD 1000


def test_sorts_and_drops_duplicates_and_nans():
    t = np.r_[np.linspace(10, 0, 300), [5.0]]          # reversed + a duplicate timestamp
    f = np.r_[1 + 1e-3 * np.arange(299) % 7, [np.nan, 1.0]][: len(t)]
    data = ("time,flux\n" + "\n".join(f"{a},{b}" for a, b in zip(t, f))).encode()
    lc = ingest.parse_light_curve("a.csv", data)
    assert np.all(np.diff(lc["time"]) > 0) and np.all(np.isfinite(lc["flux"]))


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"),
    (b"hello\nworld\n", "time and a flux"),
    (_csv(n=50), "usable points"),
    (_csv(span=0.5), "spans only"),
    (_csv(flux=np.ones(500)), "constant"),
    (b"x" * (ingest.MAX_BYTES + 1), "limit"),
])
def test_bad_files_fail_with_readable_message(data, msg):
    with pytest.raises(ingest.IngestError, match=msg):
        ingest.parse_light_curve("a.csv", data)


def test_quality_flags_are_applied():
    n = 400
    q = np.zeros(n, int); q[:250] = 1                     # first 250 cadences flagged bad
    data = ("time,flux,quality\n" + "\n".join(f"{i*0.05},{1+1e-3*(i%7)},{q[i]}" for i in range(n))).encode()
    with pytest.raises(ingest.IngestError, match="usable points"):
        ingest.parse_light_curve("a.csv", data)


def test_fits_roundtrip():
    fits = pytest.importorskip("astropy.io.fits")
    n = 400; t = np.linspace(1, 30, n)
    cols = [fits.Column(name="TIME", format="D", array=t),
            fits.Column(name="PDCSAP_FLUX", format="D", array=1 + 1e-3 * np.sin(t)),
            fits.Column(name="QUALITY", format="J", array=np.zeros(n, "i4"))]
    h0 = fits.PrimaryHDU(); h0.header["SECTOR"] = 7
    buf = io.BytesIO(); fits.HDUList([h0, fits.BinTableHDU.from_columns(cols)]).writeto(buf)
    lc = ingest.parse_light_curve("x.fits", buf.getvalue())
    assert len(lc["time"]) == n and lc["sector"] == 7


def test_save_upload_is_idempotent_and_matches_raw_format(tmp_path):
    a = ingest.save_upload("a.csv", _csv(), tmp_path)
    assert a == ingest.save_upload("b.csv", _csv(), tmp_path) and a.startswith("UPL_")
    d = np.load(tmp_path / f"{a}.npz")
    assert {"time", "pdcsap_flux", "quality", "sector", "mom_centr1"} <= set(d.files)
