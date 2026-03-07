# batch_run_gait_dataset1_rdv.py
# -*- coding: utf-8 -*-
"""
Batch runner for Dataset.

Input (given by --dataset-root):
  Kinematics/Dataset(1)/
    controle/<SUBJECT>/
      <SUBJECT>_outputData.mat
      Bare_fast_indexData.mat
      ...
    hemiparetique/<PATIENT>/RDV1..RDV5/
      <PATIENT>_<RDV>_outputData.mat
      Bare_fast_indexData.mat
      ...

Outputs (default: ./BatchOutputs):
  BatchOutputs/
    controle/<SUBJECT>/
      Summary/
        <SUBJECT>GaitSummary.csv      (ONE CSV per subject, all conditions)
        <SUBJECT>GaitParameters.csv   (NEW CSV: FG/FH/EH/WalkingSpeed)
      Figures/
        controle_<SUBJECT>_<COND>_CyclicGaitData.png
        controle_<SUBJECT>_<COND>_SwingDebug.png
        controle_<SUBJECT>_<COND>_JointAnglesSwing.png

    hemiparetique/<PATIENT>/<RDV>/
      Summary/
        <PATIENT>_<RDV>GaitSummary.csv   (ONE CSV per patient+RDV, all conditions)
        <PATIENT>_<RDV>GaitSteps.csv     (ONE CSV per patient+RDV, per-step values, all conditions)
        <PATIENT>_<RDV>GaitParameters.csv (NEW CSV: FG/FH/EH/WalkingSpeed)
      Figures/
        hemiparetique_<PATIENT>_<RDV>_<COND>_CyclicGaitData.png
        hemiparetique_<PATIENT>_<RDV>_<COND>_SwingDebug.png
        hemiparetique_<PATIENT>_<RDV>_<COND>_JointAnglesSwing.png

Summary CSV format (as requested):
  File,Parameter,Mean,Std,Unit
Where:
  - File = condition name: Bare_fast, Bare_pref, Shoe_fast, Shoe_pref
  - Parameter includes explicit: FG, FH, EH, WalkingSpeed (+ other gait parameters)

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio


# ---------------------------------------------------------------------------
# Imports from your project
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from src.gait_processing import process_gait_data
    from src.gait_functions import remove_outliers_and_compute_mean
    from src.data_loader import load_trial_from_mat, apply_sensor_reorder
except Exception:
    from gait_processing import process_gait_data
    from gait_functions import remove_outliers_and_compute_mean
    from data_loader import load_trial_from_mat, apply_sensor_reorder


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
TRIAL_NAMES = [
    "Bare_calibration",
    "Bare_fast",
    "Bare_pref",
    "Shoe_calibration",
    "Shoe_fast",
    "Shoe_pref",
]

NEW_ORDER = [7, 5, 6, 4, 3, 2, 1]

# (Condition name, calibration trial index, walking trial index)
CONDITIONS = [
    ("Bare_fast", 0, 1),
    ("Bare_pref", 0, 2),
    ("Shoe_fast", 3, 4),
    ("Shoe_pref", 3, 5),
]

RDVS = ["RDV1", "RDV2", "RDV3", "RDV4", "RDV5"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _load_indices_from_dir(folder: Path) -> List[Dict]:
    indices: List[Dict] = []
    for name in TRIAL_NAMES:
        p = folder / f"{name}_indexData.mat"
        if not p.exists():
            raise FileNotFoundError(f"Missing indexData: {p}")
        indices.append(sio.loadmat(str(p), simplify_cells=True))
    return indices


def _load_trials_from_output_mat(mat_path: Path) -> List[Dict]:
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing outputData: {mat_path}")
    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    if "output" not in mat:
        raise KeyError(f"'output' not found in {mat_path}")

    raw = mat["output"]
    trials: List[Dict] = []
    for i in range(6):
        tr = load_trial_from_mat(raw[i])
        tr = apply_sensor_reorder(tr, NEW_ORDER)
        trials.append(tr)
    return trials


def load_control_subject(subject_id: str, subject_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    trials = _load_trials_from_output_mat(subject_dir / f"{subject_id}_outputData.mat")
    indices = _load_indices_from_dir(subject_dir)
    return trials, indices


def load_hemi_patient_rdv(patient_id: str, rdv: str, rdv_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    mat_path = rdv_dir / f"{patient_id}_{rdv}_outputData.mat"
    trials = _load_trials_from_output_mat(mat_path)
    indices = _load_indices_from_dir(rdv_dir)
    return trials, indices


def list_control_subjects(control_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(control_dir.iterdir()):
        if p.is_dir() and (p / f"{p.name}_outputData.mat").exists():
            out.append(p.name)
    return out


def list_hemi_patients(hemi_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(hemi_dir.iterdir()):
        if not p.is_dir():
            continue
        if "ne pas utiliser" in p.name.lower():
            continue
        out.append(p.name)
    return out


# ---------------------------------------------------------------------------
# Plot helpers (1) cyclic mean ± std over steps
# ---------------------------------------------------------------------------
def _mean_std_over_steps(cyclic_steps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cyclic_steps is None:
        return np.array([]), np.array([])
    cyclic_steps = np.asarray(cyclic_steps)
    if cyclic_steps.size == 0:
        return np.array([]), np.array([])
    mu = np.nanmean(cyclic_steps, axis=0)
    sd = np.nanstd(cyclic_steps, axis=0, ddof=1) if cyclic_steps.shape[0] > 1 else np.zeros_like(mu)
    return mu, sd


def save_cyclic_plot(result: Dict, title: str, out_path: Path, dpi: int = 200) -> None:
    cd = result.get("CycleD", {})
    T = np.asarray(cd.get("T", np.linspace(0, 100, 100))).reshape(-1)

    rows = [
        ("Hip angle (deg)",   ("ah", "nh")),
        ("Knee angle (deg)",  ("ak", "nk")),
        ("Ankle angle (deg)", ("aa", "na")),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=16)
    axes[0, 0].set_title("Affected", fontsize=13)
    axes[0, 1].set_title("Non-paretic", fontsize=13)

    for i, (ylabel, (kL, kR)) in enumerate(rows):
        for j, key in enumerate([kL, kR]):
            ax = axes[i, j]
            cyc = np.asarray(cd.get(key, np.empty((0, 0))))
            mu, sd = _mean_std_over_steps(cyc)

            if mu.size == 0:
                ax.text(0.5, 0.5, "No steps", ha="center", va="center")
            else:
                ax.fill_between(T, mu - sd, mu + sd, alpha=0.25)
                ax.plot(T, mu, linewidth=2.5)

            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Gait cycle (%)", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot helpers (2) Swing debug plot
# ---------------------------------------------------------------------------
def save_swing_debug_plot(result: Dict, title: str, out_path: Path, dpi: int = 200) -> None:
    dbg = result.get("Debug", {})
    t = np.asarray(dbg.get("time", []))

    if t.size == 0:
        return

    s1 = dbg.get("sensor1", {})
    s2 = dbg.get("sensor2", {})

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(title, fontsize=16)

    def plot_sensor(ax_df, ax_h, sensor_dict, sensor_name: str):
        Df = np.asarray(sensor_dict.get("Df", []))
        P  = np.asarray(sensor_dict.get("P", []))
        Dfd = np.asarray(sensor_dict.get("Df_dot", []))
        Hto = np.asarray(sensor_dict.get("Hto", []))
        Hic = np.asarray(sensor_dict.get("Hic", []))

        ax_df.plot(t, Df, linewidth=1.4)
        ax_df.plot(t, P, linewidth=1.4)
        ax_df.plot(t, Dfd, linewidth=1.2)
        ax_df.set_title(f"{sensor_name}: Df, P and Df_dot", fontsize=11)
        ax_df.grid(True, alpha=0.3)

        ax_h.plot(t, Hto, linewidth=1.2)
        ax_h.plot(t, Hic, linewidth=1.2)
        ax_h.set_title(f"{sensor_name}: Hto and Hic", fontsize=11)
        ax_h.set_ylim(-0.1, 1.1)
        ax_h.grid(True, alpha=0.3)

    plot_sensor(axes[0], axes[1], s1, "Sensor 1")
    plot_sensor(axes[2], axes[3], s2, "Sensor 2")

    axes[-1].set_xlabel("Time (ms)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot helpers (3) Joint angles + swing overlay
# ---------------------------------------------------------------------------
def save_jointangles_with_swing_plot(result: Dict, title: str, out_path: Path, dpi: int = 200) -> None:
    dbg = result.get("Debug", {})
    t = np.asarray(dbg.get("time", []))
    if t.size == 0:
        return

    ang = dbg.get("angles", {})
    sw  = dbg.get("swing", {})

    ah = np.asarray(ang.get("ah", [])); ak = np.asarray(ang.get("ak", [])); aa = np.asarray(ang.get("aa", []))
    nh = np.asarray(ang.get("nh", [])); nk = np.asarray(ang.get("nk", [])); na = np.asarray(ang.get("na", []))

    P_af = np.asarray(sw.get("af", []))
    P_nf = np.asarray(sw.get("nf", []))

    if ah.size == 0 and nh.size == 0:
        return

    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(title, fontsize=16)

    pairs = [
        ("Hip joint angle (°)",   ah, nh),
        ("Knee joint angle (°)",  ak, nk),
        ("Ankle joint angle (°)", aa, na),
    ]

    for i, (ylab, L, R) in enumerate(pairs):
        axL = axes[i, 0]
        axR = axes[i, 1]

        if L.size:
            axL.plot(t, L, linewidth=1.6)
        if P_af.size:
            axL.plot(t, P_af, linewidth=1.2)
        axL.set_ylabel(ylab)
        axL.grid(True, alpha=0.3)

        if R.size:
            axR.plot(t, R, linewidth=1.6)
        if P_nf.size:
            axR.plot(t, P_nf, linewidth=1.2)
        axR.grid(True, alpha=0.3)

    axes[0, 0].set_title("Affected Limb", fontsize=12)
    axes[0, 1].set_title("Unaffected Limb", fontsize=12)
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_case_plots(out_root: Path, group: str, subject: str, session: str, condition: str, result: Dict, dpi: int) -> List[Path]:
    paths: List[Path] = []

    if group == "controle":
        folder = out_root / group / subject / "Figures"
        base = f"{group}_{subject}_{condition}"
        title = f"{subject} - {condition}"
    else:
        folder = out_root / group / subject / session / "Figures"
        base = f"{group}_{subject}_{session}_{condition}"
        title = f"{subject} {session} - {condition}"

    folder.mkdir(parents=True, exist_ok=True)

    p1 = folder / f"{base}_CyclicGaitData.png"
    save_cyclic_plot(result, title, p1, dpi=dpi)
    paths.append(p1)

    p2 = folder / f"{base}_SwingDebug.png"
    save_swing_debug_plot(result, title, p2, dpi=dpi)
    if p2.exists():
        paths.append(p2)

    p3 = folder / f"{base}_JointAnglesSwing.png"
    save_jointangles_with_swing_plot(result, title, p3, dpi=dpi)
    if p3.exists():
        paths.append(p3)

    return paths


# ---------------------------------------------------------------------------
# CSV helpers
#   - Summary rows: File,Parameter,Mean,Std,Unit  (File=Condition)
#   - Steps rows:   File,StepIndex,FG_deg,FH_deg,EH_deg,WalkingSpeed_mps
# ---------------------------------------------------------------------------
def _safe_1d(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    arr = np.asarray(x).reshape(-1)
    try:
        return arr.astype(float, copy=False)
    except Exception:
        return np.array(arr, dtype=float)


def extract_summary_rows(condition: str, result: Dict) -> List[Dict]:
    """
    Build summary rows for ONE condition.
    Must include explicit: FG, FH, EH, WalkingSpeed.
    """
    rows: List[Dict] = []
    gp = result.get("GaitParm", {})
    gc = result.get("GaitChar", {})

    def add(param: str, mean, std, unit: str):
        rows.append({
            "File": condition,
            "Parameter": param,
            "Mean": float(mean) if mean is not None else np.nan,
            "Std": float(std) if std is not None else np.nan,
            "Unit": unit
        })

    # --- Walking speed (overall) ---
    if "walkingSpeed" in gp:
        add("WalkingSpeed", gp["walkingSpeed"], np.nan, "m/s")

    # --- Explicit FG/FH/EH from affected side ---
    # FG = knee flexion peak in swing = ak/maxData_sw
    if "ak" in gc and "maxData_sw" in gc["ak"]:
        _, m, s = remove_outliers_and_compute_mean(gc["ak"]["maxData_sw"])
        add("FG", m, s, "deg")

    # FH = hip flexion peak in swing = ah/maxData_sw
    if "ah" in gc and "maxData_sw" in gc["ah"]:
        _, m, s = remove_outliers_and_compute_mean(gc["ah"]["maxData_sw"])
        add("FH", m, s, "deg")

    # EH = hip max extension in stance = ah/minData_st
    if "ah" in gc and "minData_st" in gc["ah"]:
        _, m, s = remove_outliers_and_compute_mean(gc["ah"]["minData_st"])
        add("EH", m, s, "deg")

    # --- Optional: keep other gait parameters too (useful for later IA) ---
    limb_map = [("af", "Affected"), ("nf", "Non-paretic")]
    metric_units = {
        "strideLength": "m",
        "strideTime": "s",
        "stanceDuration": "%",
        "stanceTime": "s",
        "swingTime": "s",
    }
    for limb_key, limb_label in limb_map:
        if limb_key not in gp:
            continue
        p = gp[limb_key]
        for metric, unit in metric_units.items():
            if metric not in p:
                continue
            if isinstance(p[metric], dict):
                meanv = p[metric].get("meanValue", np.nan)
                stdv  = p[metric].get("stdValue", np.nan)
            else:
                meanv, stdv = np.nan, np.nan
            add(f"GaitParm/{limb_label}/{metric}", meanv, stdv, unit)

    return rows


def extract_steps_table(condition: str, result: Dict) -> pd.DataFrame:
    """
    Build per-step numeric table for ONE condition (Affected side).
    Uses GaitChar arrays per step:
      FG_step = ak/maxData_sw
      FH_step = ah/maxData_sw
      EH_step = ah/minData_st
    """
    gp = result.get("GaitParm", {})
    gc = result.get("GaitChar", {})

    walking_speed = gp.get("walkingSpeed", np.nan)

    knee_sw = _safe_1d(gc.get("ak", {}).get("maxData_sw", []))  # FG
    hip_sw  = _safe_1d(gc.get("ah", {}).get("maxData_sw", []))  # FH
    hip_st  = _safe_1d(gc.get("ah", {}).get("minData_st", []))  # EH

    n = int(max(len(knee_sw), len(hip_sw), len(hip_st), 0))
    if n == 0:
        return pd.DataFrame(columns=["File","StepIndex","FG_deg","FH_deg","EH_deg","WalkingSpeed_mps"])

    def pad(a, n):
        if len(a) == n:
            return a
        out = np.full((n,), np.nan, dtype=float)
        out[:len(a)] = a
        return out

    knee_sw = pad(knee_sw, n)
    hip_sw  = pad(hip_sw, n)
    hip_st  = pad(hip_st, n)

    return pd.DataFrame({
        "File": [condition]*n,
        "StepIndex": np.arange(1, n+1),
        "FG_deg": knee_sw,
        "FH_deg": hip_sw,
        "EH_deg": hip_st,
        "WalkingSpeed_mps": [walking_speed]*n
    })

def extract_parameters_row(patient: str, session: str, condition: str, result: Dict) -> Dict:
    """
    Extract ONE row per patient/condition with averaged parameters.
    """

    gc = result.get("GaitChar", {})
    gp = result.get("GaitParm", {})

    FG = np.nan
    FH = np.nan
    EH = np.nan

    if "ak" in gc and "maxData_sw" in gc["ak"]:
        FG = np.nanmean(gc["ak"]["maxData_sw"])

    if "ah" in gc and "maxData_sw" in gc["ah"]:
        FH = np.nanmean(gc["ah"]["maxData_sw"])

    if "ah" in gc and "minData_st" in gc["ah"]:
        EH = np.nanmean(gc["ah"]["minData_st"])

    walking_speed = gp.get("walkingSpeed", np.nan)

    return {
        "Patient": patient,
        "Session": session,
        "Condition": condition,
        "FG_deg": FG,
        "FH_deg": FH,
        "EH_deg": EH,
        "WalkingSpeed_mps": walking_speed
    }

def save_one_csv_per_subject(out_root: Path, group: str, subject: str, session: Optional[str], rows: List[Dict]) -> Path:
    """
    Save ONE SUMMARY CSV per subject/patient (as in the manual).
    """
    if group == "controle":
        folder = out_root / group / subject / "Summary"
        name = f"{subject}GaitSummary.csv"
    else:
        assert session is not None
        folder = out_root / group / subject / session / "Summary"
        name = f"{subject}_{session}GaitSummary.csv"

    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / name
    df = pd.DataFrame(rows, columns=["File", "Parameter", "Mean", "Std", "Unit"])
    df.to_csv(csv_path, index=False)
    return csv_path


def save_parameters_csv(out_root: Path, group: str, subject: str, session: Optional[str], rows: List[Dict]) -> Path:

    if group == "controle":
        folder = out_root / group / subject / "Summary"
        name = f"{subject}GaitParameters.csv"
    else:
        folder = out_root / group / subject / session / "Summary"
        name = f"{subject}_{session}GaitParameters.csv"

    folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    path = folder / name
    df.to_csv(path, index=False)

    return path


# ---------------------------------------------------------------------------
# Runner: controls
# ---------------------------------------------------------------------------
def run_controls(control_dir: Path, out_root: Path, dpi: int, do_plots: bool) -> Tuple[List[Path], List[Path]]:
    created_summary: List[Path] = []
    created_steps: List[Path] = []

    subjects = list_control_subjects(control_dir)
    print(f"[INFO] controle subjects found: {len(subjects)}")
    all_param_rows = []
    for sid in subjects:
        sdir = control_dir / sid
        try:
            trials, indices = load_control_subject(sid, sdir)

            all_rows: List[Dict] = []
            all_steps_dfs: List[pd.DataFrame] = []

            param_rows: List[Dict] = []

            for cond_name, calib_i, walk_i in CONDITIONS:
                result = process_gait_data(trials[calib_i], indices[calib_i], trials[walk_i], indices[walk_i])

                # Summary rows
                all_rows.extend(extract_summary_rows(cond_name, result))

                # Steps table
                steps_df = extract_steps_table(cond_name, result)
                if not steps_df.empty:
                    all_steps_dfs.append(steps_df)

                # Parameter row for new CSV
                param_rows.append(extract_parameters_row(sid, "NA", cond_name, result))

                # Plots
                if do_plots:
                    _ = save_case_plots(out_root, "controle", sid, "NA", cond_name, result, dpi=dpi)

            # Save one summary CSV
            csv_path = save_one_csv_per_subject(out_root, group="controle", subject=sid, session=None, rows=all_rows)
            created_summary.append(csv_path)

            # Save one parameters CSV (FG/FH/EH/Speed)
            save_parameters_csv(out_root, "controle", sid, None, param_rows)

            print(f"[OK] controle {sid}")

        except Exception as e:
            print(f"[ERR] controle {sid}: {e}")

    return created_summary, created_steps


# ---------------------------------------------------------------------------
# Runner: hemiparetique
# ---------------------------------------------------------------------------
def run_hemiparetique(
    hemi_dir: Path,
    out_root: Path,
    dpi: int,
    do_plots: bool,
    rdvs: Optional[List[str]] = None
) -> Tuple[List[Path], List[Path]]:
    created_summary: List[Path] = []
    created_steps: List[Path] = []

    patients = list_hemi_patients(hemi_dir)
    print(f"[INFO] hemiparetique patients found: {len(patients)}")

    rdv_list = rdvs if rdvs else RDVS

    for pid in patients:
        pdir = hemi_dir / pid
        for rdv in rdv_list:
            rdv_dir = pdir / rdv
            if not rdv_dir.exists():
                continue

            mat_path = rdv_dir / f"{pid}_{rdv}_outputData.mat"
            if not mat_path.exists():
                continue

            try:
                trials, indices = load_hemi_patient_rdv(pid, rdv, rdv_dir)

                all_rows: List[Dict] = []
                all_steps_dfs: List[pd.DataFrame] = []

                param_rows: List[Dict] = []

                for cond_name, calib_i, walk_i in CONDITIONS:
                    result = process_gait_data(trials[calib_i], indices[calib_i], trials[walk_i], indices[walk_i])

                    # Summary rows
                    all_rows.extend(extract_summary_rows(cond_name, result))

                    # Steps table
                    steps_df = extract_steps_table(cond_name, result)
                    if not steps_df.empty:
                        all_steps_dfs.append(steps_df)

                    # Parameter row for new CSV
                    param_rows.append(extract_parameters_row(pid, rdv, cond_name, result))

                    # Plots
                    if do_plots:
                        _ = save_case_plots(out_root, "hemiparetique", pid, rdv, cond_name, result, dpi=dpi)

                # Save one summary CSV per patient+RDV
                csv_path = save_one_csv_per_subject(out_root, group="hemiparetique", subject=pid, session=rdv, rows=all_rows)
                created_summary.append(csv_path)

                # Save one parameters CSV (FG/FH/EH/Speed)
                save_parameters_csv(out_root, "hemiparetique", pid, rdv, param_rows)

                print(f"[OK] hemiparetique {pid} {rdv}")

            except Exception as e:
                print(f"[ERR] hemiparetique {pid} {rdv}: {e}")

    return created_summary, created_steps


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch gait processing for Dataset(1) with hemiparetique RDVs.")
    p.add_argument("--dataset-root", required=True, help='Path to ".../Kinematics/Dataset(1)"')
    p.add_argument("--group", choices=["controle", "hemiparetique", "all"], default="all")
    p.add_argument("--outdir", default=str(THIS_DIR / "BatchOutputs"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--rdvs",
        default="RDV1,RDV2,RDV3,RDV4,RDV5",
        help="Comma-separated list for hemiparetique (e.g., RDV2,RDV5)"
    )
    args, _ = p.parse_known_args(argv)
    return args


def _write_global_summary(out_root: Path, group: str, summary_paths: List[Path]) -> Optional[Path]:
    """
    Global CSV = concat ONLY summary CSV files (not steps).
    """
    dfs = []
    for pth in summary_paths:
        try:
            df = pd.read_csv(pth)
            # keep trace of origin (nice for debug)
            df.insert(0, "SourceCSV", str(pth).replace("\\", "/"))
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return None

    big = pd.concat(dfs, ignore_index=True)

    if group == "hemiparetique":
        big_path = out_root / "ALL_hemiparetique_Summary.csv"
    elif group == "controle":
        big_path = out_root / "ALL_controle_Summary.csv"
    else:
        big_path = out_root / "ALL_cases_Summary.csv"

    big.to_csv(big_path, index=False)
    return big_path


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    dataset_root = Path(args.dataset_root).resolve()
    control_dir = dataset_root / "controle"
    hemi_dir = dataset_root / "hemiparetique"

    out_root = Path(args.outdir).resolve()
    do_plots = not args.no_plots

    rdv_list = [x.strip() for x in str(args.rdvs).split(",") if x.strip()]

    all_summary_paths: List[Path] = []

    if args.group in ("controle", "all"):
        if not control_dir.exists():
            raise FileNotFoundError(f"Missing folder: {control_dir}")
        created_summary, _created_steps = run_controls(control_dir, out_root, args.dpi, do_plots)
        all_summary_paths += created_summary

    if args.group in ("hemiparetique", "all"):
        if not hemi_dir.exists():
            raise FileNotFoundError(f"Missing folder: {hemi_dir}")
        created_summary, _created_steps = run_hemiparetique(hemi_dir, out_root, args.dpi, do_plots, rdvs=rdv_list)
        all_summary_paths += created_summary

    # Global summary file
    if all_summary_paths:
        gpath = _write_global_summary(out_root, args.group, all_summary_paths)
        if gpath is not None:
            print(f"[OK] Global SUMMARY CSV: {gpath}")

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())