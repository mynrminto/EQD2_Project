"""DICOM CT + RTDOSE 読み込みと EQD2 計算ヘルパ。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom


@dataclass
class CTVolume:
    hu: np.ndarray  # (Z, Y, X)
    z_positions: list[float]
    px_x: float
    px_y: float
    origin: tuple[float, float, float]  # IPP of first slice
    rows: int
    cols: int


@dataclass
class DoseVolume:
    dose_gy: np.ndarray  # (Z, Y, X) on its own grid, Gy
    z_positions: list[float]
    px_x: float
    px_y: float
    origin: tuple[float, float, float]
    summation_type: str  # PLAN / FRACTION / BEAM
    dose_type: str       # PHYSICAL / EFFECTIVE


def find_project_root() -> Path:
    return Path(__file__).resolve().parent


def find_dicom_folder() -> Path:
    """DICOM フォルダを解決する。

    優先順:
      1. 環境変数 EQD2_DATA_FOLDER (絶対パス)
      2. プロジェクト直下の "*icom*data*" にマッチするサブフォルダ (default = 水ファントム)
    """
    env = os.environ.get("EQD2_DATA_FOLDER")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"EQD2_DATA_FOLDER={env} is not a directory")
    project = find_project_root()

    def has_dicom(folder: Path) -> bool:
        return any(folder.rglob("*.dcm"))

    # 1) 配布版の既定データ: サンプルデータ/水ファントム
    for cand in [project / "サンプルデータ" / "水ファントム",
                 project / "サンプルデータ" / "水ファントム（合成）"]:
        if cand.is_dir() and has_dicom(cand):
            return cand
    # 2) サンプルデータ/ 直下の最初の DICOM フォルダ
    sample = project / "サンプルデータ"
    if sample.is_dir():
        for n in sorted(os.listdir(sample)):
            full = sample / n
            if full.is_dir() and has_dicom(full):
                return full
    # 3) 従来の "*icom*data*" フォルダ (開発時の DICOM＿data)
    for n in sorted(os.listdir(project)):
        full = project / n
        if full.is_dir() and "icom" in n.lower() and "data" in n.lower():
            return full
    raise FileNotFoundError("DICOM data folder not found")


def list_modality(folder: Path, modality: str) -> list[Path]:
    out = []
    for f in folder.rglob("*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            if h.Modality == modality:
                out.append(f)
        except Exception:
            pass
    return out


def load_ct(folder: Path) -> CTVolume:
    files = list_modality(folder, "CT")
    files.sort(key=lambda f: float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]))
    if not files:
        raise FileNotFoundError("No CT slices found")
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    ref = headers[0]
    slope = float(getattr(ref, "RescaleSlope", 1))
    intercept = float(getattr(ref, "RescaleIntercept", 0))
    hu = np.stack([pydicom.dcmread(f).pixel_array for f in files]).astype(np.float32)
    hu = hu * slope + intercept
    px_y, px_x = map(float, ref.PixelSpacing)
    return CTVolume(
        hu=hu,
        z_positions=[float(h.ImagePositionPatient[2]) for h in headers],
        px_x=px_x, px_y=px_y,
        origin=tuple(map(float, ref.ImagePositionPatient)),
        rows=int(ref.Rows), cols=int(ref.Columns),
    )


def list_rtdose_files(folder: Path) -> list[Path]:
    """フォルダ内の RTDOSE ファイル一覧 (ファイル名でソート)。"""
    return sorted(list_modality(folder, "RTDOSE"), key=lambda f: f.name)


def load_rtdose(folder: Path, filename: str | None = None) -> DoseVolume | None:
    rd_files = list_modality(folder, "RTDOSE")
    if filename:
        rd_files = [f for f in rd_files if f.name == filename]
    if not rd_files:
        return None
    rd_files.sort(key=lambda f: f.name)
    rd = pydicom.dcmread(rd_files[0])
    arr = rd.pixel_array.astype(np.float32) * float(rd.DoseGridScaling)
    # GridFrameOffsetVector: Z offsets from first frame, relative to ImagePositionPatient[2]
    z0 = float(rd.ImagePositionPatient[2])
    offsets = [float(o) for o in rd.GridFrameOffsetVector]
    z_positions = [z0 + o for o in offsets]
    px_y, px_x = map(float, rd.PixelSpacing)
    return DoseVolume(
        dose_gy=arr,
        z_positions=z_positions,
        px_x=px_x, px_y=px_y,
        origin=tuple(map(float, rd.ImagePositionPatient)),
        summation_type=str(getattr(rd, "DoseSummationType", "PLAN")),
        dose_type=str(getattr(rd, "DoseType", "PHYSICAL")),
    )


def is_aligned_with(ct: CTVolume, dose: DoseVolume, tol: float = 0.5) -> bool:
    """CT グリッドと RTDOSE グリッドが同一かを判定 (合成データはここに収まる想定)。"""
    if ct.hu.shape != dose.dose_gy.shape:
        return False
    if abs(ct.origin[0] - dose.origin[0]) > tol or abs(ct.origin[1] - dose.origin[1]) > tol:
        return False
    for a, b in zip(ct.z_positions, dose.z_positions):
        if abs(a - b) > tol:
            return False
    return True


def _trilinear_sample(vol: np.ndarray, iz: np.ndarray, iy: np.ndarray,
                      ix: np.ndarray) -> np.ndarray:
    """純 numpy の trilinear 補間 (格子外は 0)。scipy.ndimage.map_coordinates 相当。

    vol: (nz,ny,nx)、iz/iy/ix: 補間先の分数インデックス(同一 shape にブロードキャスト済み)。
    scipy 依存を避けるため自前実装(Python バージョン非依存で動く)。
    """
    nz, ny, nx = vol.shape
    iz, iy, ix = np.broadcast_arrays(iz, iy, ix)
    z0 = np.floor(iz).astype(np.int64)
    y0 = np.floor(iy).astype(np.int64)
    x0 = np.floor(ix).astype(np.int64)
    fz, fy, fx = iz - z0, iy - y0, ix - x0

    def corner(zz, yy, xx):
        valid = (zz >= 0) & (zz < nz) & (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        v = vol[np.clip(zz, 0, nz - 1), np.clip(yy, 0, ny - 1), np.clip(xx, 0, nx - 1)]
        return np.where(valid, v, 0.0)

    c000 = corner(z0, y0, x0);     c001 = corner(z0, y0, x0 + 1)
    c010 = corner(z0, y0 + 1, x0); c011 = corner(z0, y0 + 1, x0 + 1)
    c100 = corner(z0 + 1, y0, x0);     c101 = corner(z0 + 1, y0, x0 + 1)
    c110 = corner(z0 + 1, y0 + 1, x0); c111 = corner(z0 + 1, y0 + 1, x0 + 1)
    c00 = c000 * (1 - fx) + c001 * fx
    c01 = c010 * (1 - fx) + c011 * fx
    c10 = c100 * (1 - fx) + c101 * fx
    c11 = c110 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def resample_dose_to_ct(dose: DoseVolume, ct: CTVolume) -> DoseVolume:
    """RTDOSE を CT 格子に trilinear 補間でリサンプリング。

    実臨床の RTDOSE は CT と別の解像度・原点を持つことが多い。
    DVH やオーバーレイ表示のため、CT 格子に揃える。(純 numpy 実装, scipy 不要)
    """
    nz_ct, ny_ct, nx_ct = ct.hu.shape

    # CT 各 voxel の世界座標 (mm)
    x_ct = ct.origin[0] + np.arange(nx_ct, dtype=np.float32) * ct.px_x  # (nx_ct,)
    y_ct = ct.origin[1] + np.arange(ny_ct, dtype=np.float32) * ct.px_y  # (ny_ct,)
    z_ct = np.array(ct.z_positions, dtype=np.float32)                    # (nz_ct,)

    # Dose 格子の世界座標
    dose_z = np.array(dose.z_positions, dtype=np.float32)
    # Dose 格子インデックス座標 (fractional)
    iz = (z_ct[:, None, None] - dose_z[0]) / (dose_z[1] - dose_z[0]) if len(dose_z) > 1 \
         else np.zeros((nz_ct, 1, 1), dtype=np.float32)
    iy = (y_ct[None, :, None] - dose.origin[1]) / dose.px_y
    ix = (x_ct[None, None, :] - dose.origin[0]) / dose.px_x

    resampled = _trilinear_sample(dose.dose_gy, iz, iy, ix)

    return DoseVolume(
        dose_gy=resampled.astype(np.float32),
        z_positions=list(map(float, ct.z_positions)),
        px_x=ct.px_x, px_y=ct.px_y,
        origin=ct.origin,
        summation_type=dose.summation_type,
        dose_type=dose.dose_type,
    )


def load_rtdose_aligned(folder: Path, ct: CTVolume,
                        filename: str | None = None) -> DoseVolume | None:
    """RTDOSE を読み込み、CT 格子と異なれば自動でリサンプリングして返す。"""
    dose = load_rtdose(folder, filename=filename)
    if dose is None:
        return None
    if is_aligned_with(ct, dose):
        return dose
    return resample_dose_to_ct(dose, ct)


def eqd2_map(total_dose_gy: np.ndarray, n_fractions: int, alpha_beta: float) -> np.ndarray:
    """voxel-wise EQD2 map (standard LQ). d_voxel = D_voxel / n."""
    d = total_dose_gy / float(n_fractions)
    return total_dose_gy * (d + alpha_beta) / (2.0 + alpha_beta)


def bed_map(total_dose_gy: np.ndarray, n_fractions: int, alpha_beta: float) -> np.ndarray:
    d = total_dose_gy / float(n_fractions)
    return total_dose_gy * (1.0 + d / alpha_beta)


def eqd2_lq_l(total_dose_gy: np.ndarray, n_fractions: int,
              alpha_beta: float, d_T: float = 6.0) -> np.ndarray:
    """LQ-Linear モデルによる voxel-wise EQD2 (Astrahan 2008)。

    d_voxel = D_voxel / n が d_T を超えると、LQ の傾きを保ったまま線形外挿。
    BED/n = d_T (1 + d_T/(α/β)) + (d − d_T)(1 + 2 d_T/(α/β))   (d > d_T)
    BED/n = d (1 + d/(α/β))                                       (d ≤ d_T)

    SBRT/SRS など高線量寡分割で LQ の過大評価を補正する。
    """
    d = total_dose_gy / float(n_fractions)
    ab = float(alpha_beta)
    bed_lq = total_dose_gy * (1.0 + d / ab)
    bed_lq_l = (n_fractions * d_T * (1.0 + d_T / ab)
                + n_fractions * (d - d_T) * (1.0 + 2.0 * d_T / ab))
    bed = np.where(d <= d_T, bed_lq, bed_lq_l)
    return bed / (1.0 + 2.0 / ab)


def eqd2_ir(total_dose_gy: np.ndarray, n_fractions: int, alpha_beta: float,
           ir_ratio: float = 3.0, d_c: float = 0.25) -> np.ndarray:
    """Induced Repair (IR) モデルによる voxel-wise EQD2 (Joiner, Marples ら)。

    超低線量域 (d ≲ 0.5 Gy/fx) で細胞が過敏 (HRS: hyper-radiosensitivity) になる
    現象を記述。線量増加で修復が誘導され抵抗性 (IRR) に転じ、LQ に収束する。
    LQ-L / USC が「高線量側」の補正なのに対し、IR は「低線量側」の補正。

    α_eff(d) = α_r · [1 + (α_s/α_r − 1)·exp(−d/d_c)]
    BED_IR   = D · [1 + (ir_ratio − 1)·exp(−d/d_c) + d/(α/β)]

    パラメータ:
        ir_ratio = α_s/α_r : 低線量過敏の強さ (1 で LQ と同一、典型 2〜10)
        d_c                : 過敏→抵抗の遷移線量スケール (Gy, 典型 0.2〜0.3)
    """
    d = total_dose_gy / float(n_fractions)
    ab = float(alpha_beta)
    enhance = 1.0 + (ir_ratio - 1.0) * np.exp(-d / d_c)
    bed = total_dose_gy * (enhance + d / ab)
    return bed / (1.0 + 2.0 / ab)


def usc_transition_dose(alpha: float, D0: float, Dq: float) -> float:
    """USC の遷移線量 D_T = 2·Dq / (1 − α·D0) (Park 2008)。"""
    denom = 1.0 - alpha * D0
    if denom <= 0:
        return float("inf")
    return 2.0 * Dq / denom


def eqd2_usc(total_dose_gy: np.ndarray, n_fractions: int, alpha_beta: float,
             alpha: float = 0.30, D0: float = 1.25, Dq: float = 1.8) -> np.ndarray:
    """Universal Survival Curve モデルによる voxel-wise EQD2 (Park, Timmerman ら 2008)。

    低線量域 (d ≤ D_T) は古典 LQ、高線量域 (d > D_T) は多標的(multi-target)モデルの
    終末傾き 1/D0 に切替えるハイブリッド。LQ-L が LQ の接線傾きを使うのに対し、
    USC は多標的モデルの傾きを使う点が異なる (生物物理的根拠あり)。

    パラメータ (既定値は Park 2008 NSCLC 近傍):
        alpha_beta : α/β 比 (Gy)。EQD2 換算と LQ 部に使用。
        alpha      : LQ の α 係数 (Gy^-1)。高線量域の BED 換算に必要。
        D0         : 多標的モデルの終末勾配の逆数 (Gy)。
        Dq         : 準閾値線量 (Gy) = D0·ln(n)。

    遷移線量: D_T = 2·Dq / (1 − α·D0)
    """
    d = total_dose_gy / float(n_fractions)
    ab = float(alpha_beta)
    D_T = usc_transition_dose(alpha, D0, Dq)
    bed_lq = total_dose_gy * (1.0 + d / ab)  # = n·d·(1+d/ab)
    # 高線量域: LQ(D_T までの寄与) + 多標的終末傾き
    bed_high = (n_fractions * D_T * (1.0 + D_T / ab)
                + n_fractions * (d - D_T) / (alpha * D0))
    bed = np.where(d <= D_T, bed_lq, bed_high)
    return bed / (1.0 + 2.0 / ab)


def eqd2_range_map(total_dose_gy: np.ndarray, n_fractions: int,
                    ab_low: float, ab_high: float,
                    model: str = "LQ", d_T: float = 6.0) -> dict:
    """α/β 不確実性下の EQD2 範囲 (worst-case マップ)。

    OAR (晩期反応組織) の臨床判断では低い α/β が worst-case (EQD2 高)、
    腫瘍の coverage 評価では高い α/β が worst-case (EQD2 低) になることが多い。
    """
    def f(ab):
        if model == "LQ-L":
            return eqd2_lq_l(total_dose_gy, n_fractions, ab, d_T=d_T)
        return eqd2_map(total_dose_gy, n_fractions, ab)
    e_low = f(ab_low)
    e_high = f(ab_high)
    return {
        "low_ab": e_low, "high_ab": e_high,
        "min": np.minimum(e_low, e_high),
        "max": np.maximum(e_low, e_high),
        "range": np.abs(e_high - e_low),
        "mean": (e_low + e_high) / 2.0,
    }


def course_contribution_maps(courses: list[tuple[np.ndarray, float]]) -> dict:
    """複数照射コースの累積 EQD2 と各コースの寄与度マップを計算。

    courses: [(eqd2_array_i, recovery_i), ...]
       recovery_i = 0 (完全蓄積) … 1 (完全回復)
       course_i の実効寄与 = eqd2_i × (1 − recovery_i)

    Returns:
        contributions: list[ndarray]    各コースの実効寄与 (Gy)
        cumulative:    ndarray          全コース累積 EQD2 (Gy)
        fractions:     list[ndarray]    各コースの分数寄与 (0..1, 累積>0 の voxel のみ意味あり)
        dominant:      ndarray (int8)   voxel ごとに最大寄与コースの index
        contribution_pct: ndarray       最大寄与コースの寄与 %
    """
    contribs = [eqd2 * (1.0 - rec) for eqd2, rec in courses]
    cum = np.zeros_like(contribs[0])
    for c in contribs:
        cum = cum + c
    safe_cum = np.where(cum > 1e-6, cum, 1.0)
    fractions = [c / safe_cum for c in contribs]
    stacked = np.stack(contribs, axis=0)
    dominant = np.argmax(stacked, axis=0).astype(np.int8)
    # Top contributor's percentage
    top_frac = np.max(np.stack(fractions, axis=0), axis=0)
    return {
        "contributions": contribs,
        "cumulative": cum,
        "fractions": fractions,
        "dominant": dominant,
        "top_contribution_pct": top_frac * 100.0,
    }


# ROI 名 → 推奨 α/β (Gy) のヒューリスティック。
# 文献値: Bentzen 2008, QUANTEC 2010, Joiner & van der Kogel 4th ed. などからの一般値。
ROI_AB_HEURISTIC: dict[str, float] = {
    # 晩期反応組織 (低 α/β = 高分割感受性)
    "cord": 2.0, "spinal": 2.0, "myelo": 2.0,
    "brainstem": 2.0, "brain": 2.0, "chiasm": 2.0, "optic": 2.0,
    "nerve": 2.0, "lens": 1.0,
    # 中間 α/β
    "lung": 3.0, "kidney": 3.0, "liver": 3.0, "heart": 3.0,
    "esophagus": 3.0, "rectum": 3.0,
    "bladder": 5.0, "bowel": 3.0, "stomach": 3.0, "duo": 3.0,
    "parotid": 3.0, "cochlea": 3.0,
    # 高 α/β (腫瘍・早期反応)
    "gtv": 10.0, "ctv": 10.0, "ptv": 10.0,
    "tumor": 10.0, "target": 10.0,
    "skin": 10.0, "mucosa": 10.0,
    # 前立腺は特殊 (低 α/β とする報告多数)
    "prostate": 1.5,
}


def suggest_alpha_beta(roi_name: str) -> float | None:
    """ROI 名から推奨 α/β を返す (該当キーワード無しは None)。"""
    n = roi_name.lower()
    # より長いキーワードが優先 (例: 'brainstem' は 'brain' より優先)
    for keyword in sorted(ROI_AB_HEURISTIC.keys(), key=len, reverse=True):
        if keyword in n:
            return ROI_AB_HEURISTIC[keyword]
    return None


# ---------- RTSTRUCT → 3D mask rasterization ----------

@dataclass
class StructureSet:
    rois: dict  # name -> bool ndarray (Z, Y, X) aligned with CT grid
    px_x: float
    px_y: float
    slice_thickness: float  # mm


def load_rtstruct_masks(folder: Path, ct: CTVolume) -> StructureSet | None:
    """RTSTRUCT の各 ROI を CT ボクセル格子に Rasterize する。

    PIL.ImageDraw.polygon (C実装scanline)で塗りつぶし。穴は XOR で表現
    (DICOM 規約: 入れ子の輪郭は穴)。
    """
    from PIL import Image, ImageDraw

    rs_files = list_modality(folder, "RTSTRUCT")
    if not rs_files:
        return None
    rs = pydicom.dcmread(rs_files[0])

    nz, ny, nx = ct.hu.shape
    x0, y0, _ = ct.origin
    z_positions = np.array(ct.z_positions)

    masks: dict[str, np.ndarray] = {}
    for roi, rc in zip(rs.StructureSetROISequence, rs.ROIContourSequence):
        name = roi.ROIName
        if not hasattr(rc, "ContourSequence"):
            continue
        mask = np.zeros((nz, ny, nx), dtype=bool)
        for contour in rc.ContourSequence:
            pts = np.array(contour.ContourData).reshape(-1, 3)
            if len(pts) < 3:
                continue
            z = float(pts[0, 2])
            z_idx = int(np.argmin(np.abs(z_positions - z)))
            xs_pix = (pts[:, 0] - x0) / ct.px_x
            ys_pix = (pts[:, 1] - y0) / ct.px_y
            polygon = list(zip(xs_pix.tolist(), ys_pix.tolist()))
            img = Image.new("1", (nx, ny), 0)
            ImageDraw.Draw(img).polygon(polygon, fill=1)
            inside = np.array(img, dtype=bool)
            mask[z_idx] ^= inside  # XOR handles nested-contour holes
        masks[name] = mask

    slice_thickness = float(abs(z_positions[1] - z_positions[0])) if nz > 1 else 1.0
    return StructureSet(rois=masks, px_x=ct.px_x, px_y=ct.px_y,
                        slice_thickness=slice_thickness)


# ---------- DVH ----------

def compute_dvh(dose_gy: np.ndarray, mask: np.ndarray, n_bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """累積 DVH。返り値: (dose_bins[Gy], volume_pct[0..100])。

    各 bin の dose 値 d について、ROI 内で線量 >= d を満たす voxel の割合。
    """
    if not mask.any():
        return np.array([0.0]), np.array([0.0])
    vals = dose_gy[mask]
    sorted_vals = np.sort(vals)
    max_d = float(sorted_vals[-1])
    bins = np.linspace(0, max(max_d * 1.05, 1e-6), n_bins)
    n = len(sorted_vals)
    counts_above = n - np.searchsorted(sorted_vals, bins, side="left")
    return bins, counts_above / n * 100.0


def dvh_metrics(dose_gy: np.ndarray, mask: np.ndarray,
                px_x: float, px_y: float, slice_thickness: float) -> dict:
    """DVH 由来の代表的指標。"""
    if not mask.any():
        return {"volume_cc": 0.0, "mean_Gy": 0.0, "min_Gy": 0.0, "max_Gy": 0.0,
                "D95_Gy": 0.0, "D50_Gy": 0.0, "D5_Gy": 0.0, "V20_pct": 0.0}
    vals = dose_gy[mask]
    voxel_vol_cc = px_x * px_y * slice_thickness / 1000.0  # mm³ → cc
    total_vol_cc = vals.size * voxel_vol_cc
    sorted_vals = np.sort(vals)  # ascending
    n = len(sorted_vals)
    # Dx = minimum dose received by x% of volume
    # → 累積DVHで volume_pct=x になる dose
    # = sorted_vals[(100-x)% index from bottom]
    d95 = float(sorted_vals[max(0, int(round(n * 0.05)) - 1)])
    d50 = float(sorted_vals[max(0, int(round(n * 0.50)) - 1)])
    d5 = float(sorted_vals[min(n - 1, int(round(n * 0.95)))])
    return {
        "volume_cc": float(total_vol_cc),
        "mean_Gy": float(vals.mean()),
        "min_Gy": float(vals.min()),
        "max_Gy": float(vals.max()),
        "D95_Gy": d95,
        "D50_Gy": d50,
        "D5_Gy": d5,
        "V20_pct": float(100 * (vals >= 20).sum() / n),
    }
