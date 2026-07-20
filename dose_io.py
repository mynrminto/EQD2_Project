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


def _interp_axis(vol: np.ndarray, idx: np.ndarray, axis: int) -> np.ndarray:
    """vol を axis 方向に、分数インデックス idx (1D) で線形補間 (格子外は 0)。"""
    n = vol.shape[axis]
    i0 = np.floor(idx).astype(np.intp)
    frac = (idx - i0).astype(np.float32)
    i1 = i0 + 1
    m0 = ((i0 >= 0) & (i0 < n)).astype(np.float32)
    m1 = ((i1 >= 0) & (i1 < n)).astype(np.float32)
    v0 = np.take(vol, np.clip(i0, 0, n - 1), axis=axis)
    v1 = np.take(vol, np.clip(i1, 0, n - 1), axis=axis)
    shape = [1] * vol.ndim
    shape[axis] = idx.shape[0]
    w0 = ((1.0 - frac) * m0).reshape(shape)
    w1 = (frac * m1).reshape(shape)
    return v0 * w0 + v1 * w1


def _trilinear_sample(vol: np.ndarray, iz: np.ndarray, iy: np.ndarray,
                      ix: np.ndarray) -> np.ndarray:
    """trilinear 補間 (格子外は 0)。軸ごとの 1D 線形補間を逐次適用する純 numpy 実装。

    trilinear は 3 方向 1D 線形補間のテンソル積なので、軸独立な分数インデックス
    (iz,iy,ix は各軸に沿ってのみ変化) に対しては軸ごとの逐次適用で厳密に一致する。
    8 隅を同時確保する素朴実装よりピークメモリが 1/3 以下で、クラウド無料枠(~1GB)の
    OOM を避けられる。scipy.ndimage.map_coordinates 相当(Python バージョン非依存)。
    vol: (nz,ny,nx)、iz/iy/ix: 各軸の分数インデックス (それぞれ 1D)。
    """
    out = _interp_axis(vol, iz, axis=0)   # (nz_ct, ny_dose, nx_dose)
    out = _interp_axis(out, iy, axis=1)   # (nz_ct, ny_ct,  nx_dose)
    out = _interp_axis(out, ix, axis=2)   # (nz_ct, ny_ct,  nx_ct)
    return out


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
    # Dose 格子インデックス座標 (fractional, 各軸 1D)
    iz = (z_ct - dose_z[0]) / (dose_z[1] - dose_z[0]) if len(dose_z) > 1 \
         else np.zeros(nz_ct, dtype=np.float32)
    iy = (y_ct - dose.origin[1]) / dose.px_y
    ix = (x_ct - dose.origin[0]) / dose.px_x

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


def write_eqd2_rtdose_bytes(dose_gy: np.ndarray, ct: "CTVolume", folder: Path,
                            dose_comment: str = "EQD2",
                            dose_type: str = "EFFECTIVE") -> bytes:
    """CT 格子に整列した生物学的線量 (EQD2 等) を DICOM RTDOSE のバイト列で返す。

    voxel 配列 (nz,ny,nx) を uint32 + DoseGridScaling で符号化し、幾何 (原点・
    ピクセル間隔・スライス位置) は CTVolume から、FrameOfReferenceUID /
    StudyInstanceUID / 患者情報 / ImageOrientationPatient は元 CT DICOM から
    継承する。これにより TPS で CT と同一フレームに登録できる。
    DoseType=EFFECTIVE は「生物学的(等価)線量」を意味し、物理線量と区別される。
    """
    import io
    import datetime
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian

    RT_DOSE_SOP = "1.2.840.10008.5.1.4.1.1.481.2"

    # 元 CT から UID・患者情報・向きを継承 (無ければ生成)
    ref = None
    for f in folder.rglob("*.dcm"):
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            if getattr(d, "Modality", "") == "CT":
                ref = {
                    "frame": getattr(d, "FrameOfReferenceUID", None) or generate_uid(),
                    "study": getattr(d, "StudyInstanceUID", None) or generate_uid(),
                    "name": str(getattr(d, "PatientName", "")),
                    "id": str(getattr(d, "PatientID", "")),
                    "iop": [float(x) for x in getattr(d, "ImageOrientationPatient",
                                                      [1, 0, 0, 0, 1, 0])],
                }
                break
        except Exception:
            continue
    if ref is None:
        ref = {"frame": generate_uid(), "study": generate_uid(),
               "name": "", "id": "", "iop": [1, 0, 0, 0, 1, 0]}

    n_slices, rows, cols = dose_gy.shape
    peak = max(float(np.nanmax(dose_gy)), 1e-6)
    scale = peak / (2 ** 32 - 1)
    finite = np.nan_to_num(np.asarray(dose_gy, dtype=np.float64),
                           nan=0.0, posinf=peak, neginf=0.0)
    arr_u32 = np.clip(np.rint(finite / scale), 0, 2 ** 32 - 1).astype(np.uint32)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = RT_DOSE_SOP
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("eqd2.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    now = datetime.datetime.now()
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.SOPClassUID = RT_DOSE_SOP
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTDOSE"
    ds.Manufacturer = "EQD2 Suite"
    ds.PatientName = ref["name"]
    ds.PatientID = ref["id"]
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.StudyInstanceUID = ref["study"]
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = dose_comment
    ds.StudyID = "1"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime
    ds.FrameOfReferenceUID = ref["frame"]
    ds.PositionReferenceIndicator = ""

    ds.ImagePositionPatient = [float(v) for v in ct.origin]
    ds.ImageOrientationPatient = ref["iop"]
    ds.PixelSpacing = [float(ct.px_y), float(ct.px_x)]
    ds.SliceThickness = (float(abs(ct.z_positions[1] - ct.z_positions[0]))
                         if n_slices > 1 else 1.0)

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = int(n_slices)
    ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)  # GridFrameOffsetVector
    ds.GridFrameOffsetVector = [float(z - ct.z_positions[0]) for z in ct.z_positions]
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0

    ds.DoseUnits = "GY"
    ds.DoseType = dose_type          # EFFECTIVE = 生物学的(等価)線量
    ds.DoseSummationType = "PLAN"
    ds.DoseComment = dose_comment
    ds.DoseGridScaling = scale
    ds.PixelData = arr_u32.tobytes()

    buf = io.BytesIO()
    try:
        # pydicom >= 3.0: 符号化は file_meta.TransferSyntaxUID から決定
        ds.save_as(buf, enforce_file_format=True)
    except TypeError:
        # pydicom < 3.0 (cloud/USB の 2.4 等)
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


# ---------- Clinical Goal 用の線量指標 (Eclipse の Objective 相当) ----------
def dose_to_volume_cc(dose_gy: np.ndarray, mask: np.ndarray,
                      volume_cc: float, voxel_cc: float) -> float:
    """指定体積 (cc) が受ける最小線量 D_{x cm³}。Eclipse の "D 0.3 cm³" 相当。

    ROI 内で線量の高い上位 n voxel (n = volume_cc / voxel_cc) を取り、その最小値を返す。
    """
    vals = dose_gy[mask]
    if vals.size == 0 or voxel_cc <= 0:
        return 0.0
    n_vox = max(1, min(int(round(volume_cc / voxel_cc)), vals.size))
    top = np.partition(vals, vals.size - n_vox)[vals.size - n_vox:]
    return float(top.min())


def dose_to_volume_pct(dose_gy: np.ndarray, mask: np.ndarray, pct: float) -> float:
    """体積 pct% が受ける最小線量 D_{x%} (D95 等)。"""
    vals = dose_gy[mask]
    if vals.size == 0:
        return 0.0
    k = max(1, min(int(round(vals.size * pct / 100.0)), vals.size))
    top = np.partition(vals, vals.size - k)[vals.size - k:]
    return float(top.min())


def volume_above_dose(dose_gy: np.ndarray, mask: np.ndarray,
                      gy: float, voxel_cc: float) -> tuple[float, float]:
    """線量 gy 以上を受ける体積 (cc, %) — V_xGy。"""
    vals = dose_gy[mask]
    if vals.size == 0:
        return 0.0, 0.0
    n = int((vals >= gy).sum())
    return float(n * voxel_cc), float(100.0 * n / vals.size)


# 指標名 → 説明 (UI 用)
GOAL_METRICS = {
    "Dmax":  "最大線量",
    "Dmean": "平均線量",
    "D_cc":  "指定体積(cc)が受ける最小線量 (例 D 0.3cm³)",
    "D_%":   "指定体積(%)が受ける最小線量 (例 D95%)",
    "V_Gy":  "指定線量(Gy)以上を受ける体積 (%)",
}


def evaluate_goal(dose_gy: np.ndarray, mask: np.ndarray, metric: str, param: float,
                  voxel_cc: float) -> tuple[float, str]:
    """Clinical Goal の実測値を返す。戻り値: (値, 単位)。"""
    if not mask.any():
        return 0.0, "-"
    if metric == "Dmax":
        return float(dose_gy[mask].max()), "Gy"
    if metric == "Dmean":
        return float(dose_gy[mask].mean()), "Gy"
    if metric == "D_cc":
        return dose_to_volume_cc(dose_gy, mask, param, voxel_cc), "Gy"
    if metric == "D_%":
        return dose_to_volume_pct(dose_gy, mask, param), "Gy"
    if metric == "V_Gy":
        return volume_above_dose(dose_gy, mask, param, voxel_cc)[1], "%"
    return 0.0, "-"


# ROI 名キーワード → 既定 Clinical Goal (Eclipse の Clinical Goal Template 相当)。
# (キーワード, 優先度P, 指標, パラメータ, 演算子, Objective, Variation)
# ※ 累積 EQD2 に対する一般的な文献値 (QUANTEC 等) に基づく**出発点の目安**。
#   分割・併用療法・再照射の既往・施設基準により調整が必須。
CLINICAL_GOAL_TEMPLATES: list[tuple] = [
    ("cord",      1, "D_cc",  0.03, "≤", 45.0, 50.0),
    ("spinal",    1, "D_cc",  0.03, "≤", 45.0, 50.0),
    ("myelo",     1, "D_cc",  0.03, "≤", 45.0, 50.0),
    ("brainstem", 1, "Dmax",  0.0,  "≤", 54.0, 60.0),
    ("chiasm",    1, "Dmax",  0.0,  "≤", 54.0, 56.0),
    ("optic",     1, "Dmax",  0.0,  "≤", 54.0, 56.0),
    ("esophagus", 2, "Dmax",  0.0,  "≤", 60.0, 66.0),
    ("heart",     2, "Dmean", 0.0,  "≤", 26.0, 30.0),
    ("lung",      2, "Dmean", 0.0,  "≤", 20.0, 23.0),
    ("liver",     2, "Dmean", 0.0,  "≤", 30.0, 32.0),
    ("kidney",    2, "Dmean", 0.0,  "≤", 18.0, 20.0),
    ("rectum",    2, "V_Gy",  60.0, "≤", 35.0, 40.0),
    ("bladder",   2, "V_Gy",  65.0, "≤", 50.0, 55.0),
    ("parotid",   3, "Dmean", 0.0,  "≤", 26.0, 30.0),
    ("bowel",     2, "Dmax",  0.0,  "≤", 55.0, 60.0),
    ("stomach",   2, "Dmax",  0.0,  "≤", 54.0, 60.0),
    # 標的: カバレッジ
    ("ptv",       1, "D_%",   95.0, "≥", 60.0, 57.0),
    ("ctv",       1, "D_%",   95.0, "≥", 60.0, 57.0),
    ("gtv",       1, "D_%",   95.0, "≥", 60.0, 57.0),
]


def default_clinical_goals(roi_names: list[str]) -> list[dict]:
    """ROI 名から既定の Clinical Goal を提案する (該当しない ROI は Report only)。

    返り値は UI の表に流し込める dict のリスト。値はあくまで**目安**であり、
    臨床判断は施設基準・症例背景を踏まえて行うこと。
    """
    out = []
    for name in roi_names:
        low = name.lower()
        hit = next((t for t in CLINICAL_GOAL_TEMPLATES if t[0] in low), None)
        if hit:
            _, p, metric, param, op, obj, var = hit
        else:
            p, metric, param, op, obj, var = "R", "Dmax", 0.0, "≤", 0.0, 0.0
        out.append({"P": p, "Structure": name, "指標": metric, "Param": param,
                    "演算子": op, "Objective": obj, "Variation": var})
    return out
