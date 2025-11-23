#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_from_pkl.py  —  CPU-only 精简版推理脚本
加载 best_model.pkl 并对 CSV 或 JSON 批量推理。
自动处理 Group → Group_1..Group_7 one-hot。
"""

import os, sys, json, argparse, datetime, warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------- TinyMLP（与训练时一致） -------------------
class TinyMLP(nn.Module):
    def __init__(self, dim: int, activation: str = "gelu", dropout: float = 0.2):
        super().__init__()
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ------------------- 工具函数 -------------------
def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"缺少必需列：{c}")
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[cols].isna().any().any():
        bad = df[cols].columns[df[cols].isna().any()].tolist()
        raise ValueError(f"以下列存在非数值或缺失：{bad}")

def auto_expand_group_onehot(df: pd.DataFrame, feature_cols):
    """若 feature_cols 需要 Group_1..7，但输入仅有 'Group'(1..7)，自动展开。"""
    need_groups = [c for c in feature_cols if c.startswith("Group_")]
    if not need_groups:
        return df
    have_all = all(c in df.columns for c in need_groups)
    if have_all:
        df[need_groups] = df[need_groups].apply(pd.to_numeric, errors="coerce").fillna(0).clip(0,1)
        return df
    if "Group" in df.columns:
        g = pd.to_numeric(df["Group"], errors="coerce").astype("Int64")
        for k in range(1, 8):
            df[f"Group_{k}"] = (g == k).astype(float)
        return df
    raise ValueError(f"缺少 Group_1..7，也没有 Group 列可展开。")

# ------------------- 主流程 -------------------
def main():
    ap = argparse.ArgumentParser(description="Infer from best_model.pkl (CPU only)")
    ap.add_argument("--pkl", required=True, help="best_model.pkl 路径")
    ap.add_argument("--json", help="输入 JSON 文件（{'records': [ {...}, {...} ]}）")
    ap.add_argument("--csv", help="输入 CSV 文件")
    ap.add_argument("--out", default=None, help="输出预测 CSV（默认同目录）")
    args = ap.parse_args()

    if not os.path.exists(args.pkl):
        raise FileNotFoundError(args.pkl)

    # ===== 1. 读取模型 =====
    bundle = joblib.load(args.pkl)
    feature_cols = bundle["feature_cols"]
    target_col   = bundle.get("target_col", "VFA_change")
    scaler = bundle["scaler"]
    Kmat   = np.asarray(bundle["projection"]["Kmat"], dtype=np.float32)
    x_mean = np.asarray(bundle["standardization"]["x_mean"], dtype=np.float32)
    x_std  = np.asarray(bundle["standardization"]["x_std"], dtype=np.float32)
    y_mean = float(bundle["standardization"]["y_mean"])
    y_std  = float(bundle["standardization"]["y_std"])
    spec   = bundle["model_spec"]
    init_args  = dict(spec.get("init_args", {}))
    state_dict = spec["state_dict"]

    # ===== 2. 加载输入 =====
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "records" not in data or not isinstance(data["records"], list):
            raise ValueError("JSON 应为 {'records':[ {...}, {...} ]}")
        df = pd.DataFrame(data["records"])
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        raise ValueError("必须提供 --json 或 --csv")

    # ===== 3. 自动展开 Group one-hot =====
    df = auto_expand_group_onehot(df, feature_cols)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入缺少列: {missing}")
    X_df = df[feature_cols].copy()
    ensure_numeric(X_df, feature_cols)

    # ===== 4. Scaler → 投影 → 标准化 =====
    X = scaler.transform(X_df.to_numpy(dtype=np.float32))
    Z = X @ Kmat
    Zn = (Z - x_mean) / np.clip(x_std, 1e-6, None)

    # ===== 5. TinyMLP 推理（CPU） =====
    K = Kmat.shape[1]
    init_args["dim"] = K
    model = TinyMLP(**init_args)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    Zt = torch.as_tensor(Zn, dtype=torch.float32)
    with torch.no_grad():
        y_pred_n = model(Zt).detach().numpy()
        y_pred = y_pred_n * y_std + y_mean

    # ===== 6. 输出结果 =====
    out_df = pd.DataFrame({"Pred": y_pred.astype(np.float64)})
    if target_col in df.columns:
        y_true = pd.to_numeric(df[target_col], errors="coerce").to_numpy()
        mask = ~np.isnan(y_true)
        if mask.sum() > 1:
            R2 = float(r2_score(y_true[mask], y_pred[mask]))
            RMSE = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            MAE = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            print(f"[Eval] R2={R2:.6f} | RMSE={RMSE:.6f} | MAE={MAE:.6f}")
        out_df["True"] = df[target_col].values

    out_path = args.out
    if not out_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(args.pkl) or "."
        out_path = os.path.join(base_dir, f"infer_preds_{ts}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False, float_format="%.10g")

    print(f"[Saved] {out_path}")
    print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
