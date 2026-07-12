"""TCIA から RT データセットをダウンロードするヘルパ。

デフォルトでは Pancreatic-CT-CBCT-SEG / Pancreas-CT-CB_001 の
プランニング CT + RTSTRUCT + RTDOSE をダウンロードする。

usage:
    python3 download_tcia.py                            # default患者をDL
    python3 download_tcia.py --patient Pancreas-CT-CB_005

ダウンロード後 Streamlit を切り替えるには:
    EQD2_DATA_FOLDER="$(pwd)/TCIA_data/Pancreas-CT-CB_001" \\
      python3 -m streamlit run app.py
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
HDR = {"Accept": "application/json"}

PROJECT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT / "TCIA_data"


def get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def download_series(uid: str, out_dir: Path, label: str) -> tuple[int, float]:
    """シリーズを out_dir 配下の固有サブフォルダに展開する (ファイル名衝突防止)。"""
    url = f"{API}/getImage?SeriesInstanceUID={uid}"
    sub = out_dir / uid[-12:]  # UID 末尾12文字をフォルダ名に
    sub.mkdir(parents=True, exist_ok=True)
    print(f"  → {label}")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=900) as r:
        data = r.read()
    size_mb = len(data) / (1024 ** 2)
    n_files = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.filename.startswith("__") or info.filename.endswith(".DS_Store"):
                continue
            zf.extract(info, sub)
            n_files += 1
    print(f"     extracted {n_files} file(s), {size_mb:.1f} MB ({time.time()-t0:.1f}s) → {sub.name}/")
    return n_files, size_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="Pancreatic-CT-CBCT-SEG")
    ap.add_argument("--patient", default="Pancreas-CT-CB_001")
    ap.add_argument("--ct-desc", default="PANCREAS DI",
                    help="ダウンロードしたい CT の SeriesDescription 部分文字列")
    ap.add_argument("--struct-desc", default="BSPC",
                    help="同じく RTSTRUCT 用 (PC = planning CT 用構造体)")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.patient
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Collection: {args.collection}")
    print(f"Patient   : {args.patient}")
    print(f"Output dir: {out_dir}\n")

    print("Listing series…")
    series = get_json(f"{API}/getSeries?Collection={args.collection}&PatientID={args.patient}")
    print(f"  found {len(series)} series total")

    plan = []
    # CT (planning)
    cts = [s for s in series if s["Modality"] == "CT" and args.ct_desc in s.get("SeriesDescription", "")]
    if not cts:
        # Fallback: pick the largest CT series
        cts = sorted([s for s in series if s["Modality"] == "CT"],
                     key=lambda s: -int(s.get("ImageCount", 0)))[:1]
    plan.extend(cts[:1])
    # RTSTRUCT (planning)
    rss = [s for s in series if s["Modality"] == "RTSTRUCT" and args.struct_desc in s.get("SeriesDescription", "")]
    plan.extend(rss[:1] if rss else [s for s in series if s["Modality"] == "RTSTRUCT"][:1])
    # RTDOSE (all)
    plan.extend([s for s in series if s["Modality"] == "RTDOSE"])

    print(f"\nDownload plan ({len(plan)} series):")
    for s in plan:
        print(f"  {s['Modality']:>9s}  n={s.get('ImageCount', '?'):>4}  {s.get('SeriesDescription', '')}")

    total_files, total_mb = 0, 0.0
    for s in plan:
        label = f"{s['Modality']} ({s.get('SeriesDescription', '?')[:40]})"
        n, mb = download_series(s["SeriesInstanceUID"], out_dir, label)
        total_files += n
        total_mb += mb

    print(f"\nDone. {total_files} files, ~{total_mb:.1f} MB at {out_dir}")
    print(f"\nUse in app:")
    print(f"  EQD2_DATA_FOLDER='{out_dir}' python3 -m streamlit run app.py")


if __name__ == "__main__":
    main()
