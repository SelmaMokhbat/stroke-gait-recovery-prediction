
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path


TRIAL_NAMES = [
    'Bare_calibration',
    'Bare_fast',
    'Bare_pref',
    'Shoe_calibration',
    'Shoe_fast',
    'Shoe_pref'
]

SENSOR_REORDER = [7, 5, 6, 4, 3, 2, 1]  # Maps device order to body position order


def load_sensor_from_txt(txt_file):
    """Load one sensor from a .txt file"""
    df = pd.read_csv(txt_file, sep='\t', comment='/')
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[1:]  # Skip first row (matches outputData.mat behavior)

    return {
        'quat':  df[['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']].values.astype(np.float64),
        'acc':   df[['Acc_X', 'Acc_Y', 'Acc_Z']].values.astype(np.float64),
        'gyr':   df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values.astype(np.float64),
        'mag':   df[['Mag_X', 'Mag_Y', 'Mag_Z']].values.astype(np.float64),
        'euler': df[['Roll', 'Pitch', 'Yaw']].values.astype(np.float64)
    }


def load_trial_from_txt(trial_dir, trial_name):
    """Load one trial from a directory of .txt files"""
    trial_dir = Path(trial_dir)
    txt_files = sorted(trial_dir.glob('*.txt'))

    if len(txt_files) != 7:
        raise ValueError(
            f"Expected 7 sensor files in {trial_dir}, found {len(txt_files)}"
        )

    cal_data = [load_sensor_from_txt(f) for f in txt_files]
    return {'Name': trial_name, 'cal': cal_data}


def load_trial_from_mat(mat_trial):
    """Convert a trial from outputData.mat format to standard dict"""
    cal_data = []
    for sensor in mat_trial['cal']:
        cal_data.append({
            'quat':  np.array(sensor['quat'], dtype=np.float64),
            'acc':   np.array(sensor['acc'],  dtype=np.float64),
            'gyr':   np.array(sensor['gyr'],  dtype=np.float64),
            'mag':   np.array(sensor['mag'],  dtype=np.float64),
            'euler': np.array(sensor['euler'], dtype=np.float64)
        })
    return {'Name': mat_trial['Name'], 'cal': cal_data}


def apply_sensor_reorder(trial, new_order=SENSOR_REORDER):
    """
    Reorder sensors from device order to body position order.

    Body positions after reorder:
    [0] waist_IMU
    [1] Non-affected Thigh
    [2] Non-affected Shank
    [3] Non-affected Foot
    [4] Affected Thigh
    [5] Affected Shank
    [6] Affected Foot
    """
    reordered_cal = [trial['cal'][i - 1] for i in new_order]
    return {'Name': trial['Name'], 'cal': reordered_cal}


def load_patient_data(patient_id, rdv='RDV1', base_path=None):

    if base_path is None:
        base_path = Path(__file__).parent.parent / 'Kinematics'

    mat_path = base_path / 'Dataset' / 'hemiparetic' / patient_id / rdv / f'{patient_id}_{rdv}_outputData.mat'
    txt_base = base_path / 'Dataset' / 'hemiparetique' / patient_id / rdv / 'Data'

    # Try outputData.mat first
    if mat_path.exists():
        mat_data = sio.loadmat(str(mat_path), simplify_cells=True)
        raw_output = mat_data['output']
        trials = [load_trial_from_mat(raw_output[i]) for i in range(6)]
        source = 'mat'

    # Fall back to raw txt files
    elif txt_base.exists():
        txt_complete = all(
            len(list((txt_base / t).glob('*.txt'))) == 7
            for t in TRIAL_NAMES
            if (txt_base / t).exists()
        )

        missing = [t for t in TRIAL_NAMES if not (txt_base / t).exists() or
                   len(list((txt_base / t).glob('*.txt'))) != 7]
        if missing:
            raise FileNotFoundError(
                f"Incomplete txt data for {patient_id}: missing or incomplete trials {missing}"
            )

        trials = [load_trial_from_txt(txt_base / t, t) for t in TRIAL_NAMES]
        source = 'txt'

    else:
        raise FileNotFoundError(
            f"No data found for {patient_id} {rdv}: "
            f"no outputData.mat and no txt files"
        )

    # Apply sensor reorder (device order -> body position order)
    re_output = [apply_sensor_reorder(t) for t in trials]

    return re_output, source


def load_index_files(patient_id, rdv='RDV1', base_path=None):

    if base_path is None:
        base_path = Path(__file__).parent.parent / 'Kinematics'

    idx_dir = base_path / 'Dataset' / 'hemiparetic' / patient_id / rdv
    indices = {}

    for trial in TRIAL_NAMES:
        idx_file = idx_dir / f'{trial}_indexData.mat'
        if idx_file.exists():
            indices[trial] = sio.loadmat(str(idx_file), simplify_cells=True)
        else:
            indices[trial] = None

    return indices


def load_control_patient_data(patient_id, base_path=None):

    if base_path is None:
        base_path = Path(__file__).parent.parent / 'Kinematics'

    mat_path = base_path / 'Dataset' / 'controle' / patient_id / f'{patient_id}_outputData.mat'

    if not mat_path.exists():
        raise FileNotFoundError(f'No outputData.mat found for {patient_id}: {mat_path}')

    mat_data = sio.loadmat(str(mat_path), simplify_cells=True)
    raw_output = mat_data['output']
    trials = [load_trial_from_mat(raw_output[i]) for i in range(6)]
    re_output = [apply_sensor_reorder(t) for t in trials]
    return re_output


def load_control_index_files(patient_id, base_path=None):

    if base_path is None:
        base_path = Path(__file__).parent.parent / 'Kinematics'

    idx_dir = base_path / 'Dataset' / 'controle' / patient_id
    indices = []
    for trial in TRIAL_NAMES:
        idx_file = idx_dir / f'{trial}_indexData.mat'
        if not idx_file.exists():
            raise FileNotFoundError(f'Missing index file: {idx_file}')
        indices.append(sio.loadmat(str(idx_file), simplify_cells=True))
    return indices


def check_patient_availability(patient_id, rdv='RDV1', base_path=None):

    if base_path is None:
        base_path = Path(__file__).parent.parent / 'Kinematics'

    mat_path = base_path / 'Dataset' / 'hemiparetic' / patient_id / rdv / f'{patient_id}_{rdv}_outputData.mat'
    txt_base = base_path / 'Dataset' / 'hemiparetique' / patient_id / rdv / 'Data'
    idx_dir = base_path / 'Dataset' / 'hemiparetic' / patient_id / rdv

    has_mat = mat_path.exists()
    has_txt = txt_base.exists()
    has_full_txt = has_txt and all(
        len(list((txt_base / t).glob('*.txt'))) == 7
        for t in TRIAL_NAMES
    )
    has_indices = idx_dir.exists()

    return {
        'has_mat': has_mat,
        'has_txt': has_txt,
        'has_full_txt': has_full_txt,
        'has_indices': has_indices,
        'processable': has_indices and (has_mat or has_full_txt),
        'source': 'mat' if has_mat else ('txt' if has_full_txt else 'none')
    }
