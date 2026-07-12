"""CT に整合した合成 RTDOSE を生成する開発用スクリプト。

EQD2 voxel-wise 計算/可視化を実 RTDOSE 無しで進めるため、
水ファントム ROI の中心に楕円体ガウシアンの線量を置く。

usage:
    python3 synth_dose.py [--total 50] [--fractions 25] [--sigma 60]
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian


RT_DOSE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.2"


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here


def find_dicom_folder(project: Path) -> Path:
    for n in os.listdir(project):
        if "icom" in n.lower() and "data" in n.lower():
            return project / n
    raise FileNotFoundError("DICOM data folder not found")


def load_ct_volume(folder: Path):
    files = sorted(
        [str(f) for f in folder.rglob("*.dcm")
         if pydicom.dcmread(f, stop_before_pixels=True).Modality == "CT"],
        key=lambda f: float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]),
    )
    if not files:
        raise FileNotFoundError(f"No CT files in {folder}")
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    ref = headers[0]
    rows, cols = int(ref.Rows), int(ref.Columns)
    px_y, px_x = map(float, ref.PixelSpacing)
    z_positions = [float(h.ImagePositionPatient[2]) for h in headers]
    # CT volume origin = first slice's ImagePositionPatient
    ipp0 = list(map(float, ref.ImagePositionPatient))
    iop = list(map(float, ref.ImageOrientationPatient))
    return {
        "headers": headers,
        "rows": rows, "cols": cols,
        "px_x": px_x, "px_y": px_y,
        "z_positions": z_positions,
        "ipp0": ipp0,
        "iop": iop,
        "frame_uid": ref.FrameOfReferenceUID,
        "patient_id": ref.PatientID,
        "patient_name": str(ref.PatientName),
        "study_uid": ref.StudyInstanceUID,
    }


def get_water_roi_center(folder: Path, fallback_center=(0.0, 0.0, 0.0)):
    rs_files = [f for f in folder.rglob("*.dcm")
                if pydicom.dcmread(f, stop_before_pixels=True).Modality == "RTSTRUCT"]
    if not rs_files:
        return fallback_center
    rs = pydicom.dcmread(rs_files[0])
    for roi, rc in zip(rs.StructureSetROISequence, rs.ROIContourSequence):
        if roi.ROIName.lower() == "water" and hasattr(rc, "ContourSequence"):
            pts = np.vstack([np.array(c.ContourData).reshape(-1, 3) for c in rc.ContourSequence])
            return tuple(pts.mean(axis=0).tolist())
    return fallback_center


def build_dose_grid(ct, center_xyz, total_dose_gy, sigma_xyz):
    """Create a 3D dose grid (Gy) on the CT voxel grid.

    sigma_xyz: tuple(sx, sy, sz) in mm — 楕円体ガウシアン用。等方なら同値を渡す。
    Shape: (Z, Y, X) = (n_slices, rows, cols)
    """
    rows, cols = ct["rows"], ct["cols"]
    x0, y0, _ = ct["ipp0"]
    xs = x0 + np.arange(cols) * ct["px_x"]
    ys = y0 + np.arange(rows) * ct["px_y"]
    zs = np.array(ct["z_positions"])

    cx, cy, cz = center_xyz
    sx, sy, sz = sigma_xyz
    dx = (xs[None, None, :] - cx) / sx
    dy = (ys[None, :, None] - cy) / sy
    dz = (zs[:, None, None] - cz) / sz
    r2 = dx ** 2 + dy ** 2 + dz ** 2
    dose = total_dose_gy * np.exp(-0.5 * r2)
    return dose.astype(np.float32)


def write_rtdose(dose_gy: np.ndarray, ct, out_path: Path, total_dose_gy: float,
                 series_desc: str = "synthetic"):
    """Write a minimal valid RTDOSE DICOM file aligned with the CT."""
    n_slices, rows, cols = dose_gy.shape

    # Encode as uint32 with DoseGridScaling
    peak = max(float(dose_gy.max()), 1e-6)
    scale = peak / (2**32 - 1)
    arr_u32 = np.clip(dose_gy / scale, 0, 2**32 - 1).astype(np.uint32)

    # File meta
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = RT_DOSE_SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128)

    now = datetime.datetime.now()
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.SOPClassUID = RT_DOSE_SOP_CLASS_UID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTDOSE"
    ds.Manufacturer = "EQD2_Project (synthetic)"
    ds.PatientName = ct["patient_name"]
    ds.PatientID = ct["patient_id"]
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.StudyInstanceUID = ct["study_uid"]
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = series_desc
    ds.StudyID = "1"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime
    ds.FrameOfReferenceUID = ct["frame_uid"]
    ds.PositionReferenceIndicator = ""

    # Image geometry — match CT origin (X0, Y0) and z of first slice
    ds.ImagePositionPatient = list(ct["ipp0"])
    ds.ImageOrientationPatient = list(ct["iop"])
    ds.PixelSpacing = [ct["px_y"], ct["px_x"]]
    ds.SliceThickness = float(abs(ct["z_positions"][1] - ct["z_positions"][0])) if n_slices > 1 else 1.0

    # Dose grid
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = int(n_slices)
    ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)  # GridFrameOffsetVector
    ds.GridFrameOffsetVector = [z - ct["z_positions"][0] for z in ct["z_positions"]]
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0  # unsigned

    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.DoseGridScaling = scale

    ds.PixelData = arr_u32.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(out_path), write_like_original=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=float, default=50.0, help="処方総線量 Gy (default 50)")
    ap.add_argument("--fractions", type=int, default=25, help="分割回数 (default 25)")
    ap.add_argument("--sigma", type=float, default=60.0, help="ガウシアンσ mm (等方時)")
    ap.add_argument("--sigma-x", type=float, default=None, help="X方向σ mm (省略時 --sigma)")
    ap.add_argument("--sigma-y", type=float, default=None, help="Y方向σ mm (省略時 --sigma)")
    ap.add_argument("--sigma-z", type=float, default=None, help="Z方向σ mm (省略時 --sigma)")
    ap.add_argument("--offset-x", type=float, default=0.0, help="Water ROI 中心からのXオフセット mm")
    ap.add_argument("--offset-y", type=float, default=0.0, help="Water ROI 中心からのYオフセット mm")
    ap.add_argument("--offset-z", type=float, default=0.0, help="Water ROI 中心からのZオフセット mm")
    ap.add_argument("--peak", nargs=7, type=float, action="append",
                    metavar=("X", "Y", "Z", "TOTAL", "SX", "SY", "SZ"),
                    help="追加ピーク (オフセットXYZ mm, 総線量 Gy, σ XYZ mm)。複数指定可。"
                         "指定時は --total/--sigma/--offset-* は無視される。")
    ap.add_argument("--out", type=str, default="RD.synthetic.dcm", help="出力ファイル名")
    args = ap.parse_args()

    project = find_project_root()
    folder = find_dicom_folder(project)
    print(f"Project: {project}")
    print(f"DICOM folder: {folder}")

    ct = load_ct_volume(folder)
    print(f"CT volume: {len(ct['z_positions'])} slices, {ct['rows']}x{ct['cols']}, "
          f"px={ct['px_x']:.3f}mm, Z from {ct['z_positions'][0]:.1f} to {ct['z_positions'][-1]:.1f}")

    base_center = get_water_roi_center(folder)
    print(f"Water ROI center (mm): X={base_center[0]:.1f}, Y={base_center[1]:.1f}, Z={base_center[2]:.1f}")

    # Build list of peaks
    if args.peak:
        peaks = []
        for p in args.peak:
            ox, oy, oz, total, sx, sy, sz = p
            peaks.append((
                (base_center[0] + ox, base_center[1] + oy, base_center[2] + oz),
                float(total),
                (float(sx), float(sy), float(sz)),
            ))
    else:
        sx = args.sigma_x if args.sigma_x is not None else args.sigma
        sy = args.sigma_y if args.sigma_y is not None else args.sigma
        sz = args.sigma_z if args.sigma_z is not None else args.sigma
        center = (
            base_center[0] + args.offset_x,
            base_center[1] + args.offset_y,
            base_center[2] + args.offset_z,
        )
        peaks = [(center, float(args.total), (sx, sy, sz))]

    print(f"Total peaks: {len(peaks)}")
    nz = len(ct["z_positions"])
    dose = np.zeros((nz, ct["rows"], ct["cols"]), dtype=np.float32)
    grand_total = 0.0
    for i, (c, total, sigmas) in enumerate(peaks):
        contrib = build_dose_grid(ct, c, total, sigmas)
        dose += contrib
        grand_total += total
        print(f"  peak#{i+1}: center=({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f}), "
              f"total={total} Gy, σ=({sigmas[0]:.0f}, {sigmas[1]:.0f}, {sigmas[2]:.0f})mm")

    print(f"Fractions: {args.fractions}  (=> peak d = {dose.max()/args.fractions:.2f} Gy/fx)")
    print(f"Combined: shape={dose.shape}, peak={dose.max():.2f} Gy, "
          f"99%ile={np.percentile(dose, 99):.2f} Gy")

    out_path = folder / args.out
    series_desc = args.out.replace("RD.", "").replace(".dcm", "")
    write_rtdose(dose, ct, out_path, grand_total, series_desc=series_desc)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote: {out_path} ({size_mb:.1f} MB)")

    # Verification roundtrip
    rd = pydicom.dcmread(str(out_path))
    arr = rd.pixel_array * rd.DoseGridScaling
    print(f"Roundtrip check: shape={arr.shape}, peak={arr.max():.2f} Gy "
          f"({'OK' if abs(arr.max() - dose.max()) < 0.01 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
