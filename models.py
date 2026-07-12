"""生物学的線量モデルの一元管理。

全ページが共通の「モデル選択 + α/β + 分割数」を session_state 経由で共有する。
計算式の本体は dose_io.py (numpy 版、スカラーにも使える) を再利用。
"""
from __future__ import annotations

import streamlit as st

from dose_io import (
    eqd2_map, eqd2_lq_l, eqd2_usc, eqd2_ir, usc_transition_dose,
)

MODELS = ["古典LQ", "LQ-L (Astrahan)", "USC (Park/Timmerman)", "IR (Induced Repair)"]

MODEL_COLORS = {
    "古典LQ": "#ef4444",
    "LQ-L (Astrahan)": "#3b82f6",
    "USC (Park/Timmerman)": "#22c55e",
    "IR (Induced Repair)": "#f59e0b",
}

MODEL_HELP = {
    "古典LQ": "標準 Linear-Quadratic。8 Gy/fx 超で過大評価。",
    "LQ-L (Astrahan)": "d_T 以上を線形外挿し高線量を補正 (Astrahan 2008)。",
    "USC (Park/Timmerman)": "多標的モデルの終末傾きで高線量補正 (Park/Timmerman 2008)。",
    "IR (Induced Repair)": "低線量過敏 HRS/IRR を補正 (Joiner ら)。低線量裾野で効く。",
}

ALPHA_BETA_OPTIONS = [1.0, 1.5, 2.0, 3.0, 10.0]
ALPHA_BETA_HINT = {
    1.0: "1.0  脊髄など晩期反応",
    1.5: "1.5  晩期反応",
    2.0: "2.0  晩期反応 (一般)",
    3.0: "3.0  肺・腎・消化管など",
    10.0: "10.0 腫瘍・早期反応",
}


def default_params() -> dict:
    return {"d_T": 6.0, "alpha": 0.30, "D0": 1.25, "Dq": 1.8,
            "ir_ratio": 3.0, "d_c": 0.25}


def compute_eqd2(model: str, total_dose, dose_per_fx, alpha_beta: float,
                 params: dict):
    """EQD2 を計算 (スカラー / numpy 配列 両対応)。

    dose_per_fx は voxel ごとに与える場合は配列、処方計算ではスカラー。
    dose_io の関数は (total_dose, n_fractions, ab, ...) を取るので、
    n = total_dose / dose_per_fx で換算して呼ぶ。
    """
    import numpy as np
    td = np.asarray(total_dose, dtype=float)
    d = np.asarray(dose_per_fx, dtype=float)
    n = np.divide(td, d, out=np.ones_like(td), where=d > 0)
    # dose_io の関数は n を整数前提だが、式は連続なので float n でも正しい
    if model == "LQ-L (Astrahan)":
        out = eqd2_lq_l(td, n, alpha_beta, d_T=params["d_T"])
    elif model == "USC (Park/Timmerman)":
        out = eqd2_usc(td, n, alpha_beta,
                       alpha=params["alpha"], D0=params["D0"], Dq=params["Dq"])
    elif model == "IR (Induced Repair)":
        out = eqd2_ir(td, n, alpha_beta,
                      ir_ratio=params["ir_ratio"], d_c=params["d_c"])
    else:
        out = eqd2_map(td, n, alpha_beta)
    return float(out) if np.isscalar(total_dose) or np.ndim(total_dose) == 0 else out


def eqd2_volume(model: str, dose_vol, n_fractions: int, alpha_beta: float,
                params: dict):
    """RTDOSE ボリューム (総線量, Gy) を voxel-wise EQD2 に変換。"""
    if model == "LQ-L (Astrahan)":
        return eqd2_lq_l(dose_vol, n_fractions, alpha_beta, d_T=params["d_T"])
    if model == "USC (Park/Timmerman)":
        return eqd2_usc(dose_vol, n_fractions, alpha_beta,
                        alpha=params["alpha"], D0=params["D0"], Dq=params["Dq"])
    if model == "IR (Induced Repair)":
        return eqd2_ir(dose_vol, n_fractions, alpha_beta,
                       ir_ratio=params["ir_ratio"], d_c=params["d_c"])
    return eqd2_map(dose_vol, n_fractions, alpha_beta)


def model_picker(container=None, key_prefix: str = "global") -> tuple[str, dict]:
    """サイドバーに「モデル + パラメータ」を描画し session_state に保存。

    全ページの冒頭で呼ぶと、選択がページ間で共有される。
    返り値: (model, params)
    """
    c = container if container is not None else st.sidebar
    c.markdown("### 🧬 生物学的モデル")

    if "model" not in st.session_state:
        st.session_state.model = MODELS[0]
    if "model_params" not in st.session_state:
        st.session_state.model_params = default_params()

    model = c.selectbox(
        "モデル", MODELS, index=MODELS.index(st.session_state.model),
        key=f"{key_prefix}_model",
        help="低線量=IR、中線量=全モデル一致、高線量=LQ-L/USC が LQ の過大評価を補正",
    )
    st.session_state.model = model
    c.caption(MODEL_HELP[model])

    params = dict(st.session_state.model_params)
    if model == "LQ-L (Astrahan)":
        params["d_T"] = c.number_input("遷移線量 d_T (Gy/fx)", 1.0, 20.0,
                                       float(params["d_T"]), 0.5, key=f"{key_prefix}_dT")
    elif model == "USC (Park/Timmerman)":
        params["alpha"] = c.number_input("α (Gy⁻¹)", 0.05, 1.0,
                                         float(params["alpha"]), 0.01, key=f"{key_prefix}_a")
        params["D0"] = c.number_input("D0 (Gy)", 0.5, 3.0,
                                      float(params["D0"]), 0.05, key=f"{key_prefix}_d0")
        params["Dq"] = c.number_input("Dq (Gy)", 0.0, 5.0,
                                      float(params["Dq"]), 0.1, key=f"{key_prefix}_dq")
        c.caption(f"→ 遷移線量 D_T = {usc_transition_dose(params['alpha'], params['D0'], params['Dq']):.2f} Gy/fx")
    elif model == "IR (Induced Repair)":
        params["ir_ratio"] = c.number_input("α_s/α_r 比", 1.0, 15.0,
                                            float(params["ir_ratio"]), 0.5, key=f"{key_prefix}_irr")
        params["d_c"] = c.number_input("遷移 d_c (Gy)", 0.05, 1.0,
                                      float(params["d_c"]), 0.05, key=f"{key_prefix}_dc")
    st.session_state.model_params = params
    return model, params
