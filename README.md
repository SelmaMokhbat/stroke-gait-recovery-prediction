# stroke-gait-recovery-prediction

This repository contains the code and results for a project focused on **predicting functional gait recovery after stroke using wearable IMU sensors and machine learning**.

The objective of the project is to analyze gait parameters extracted from inertial sensors and evaluate whether participation in a rehabilitation program (**GSRC**) influences recovery trajectories between two clinical visits.

The project includes:

- a signal processing pipeline for extracting gait parameters from IMU data  
- a longitudinal dataset construction  
- a machine learning model to predict recovery  
- statistical comparisons between rehabilitation groups  

---

# Project Overview

Patients performed walking trials recorded with **7 IMU sensors** placed on the body.

The dataset includes measurements collected at multiple clinical visits (RDV).

In this study, the **longitudinal analysis is performed between:**

- **RDV2 (baseline)**
- **RDV5 (follow-up)**

Recovery variables are computed as the difference between these two visits.

Example:

ΔV = WalkingSpeed(RDV5) − WalkingSpeed(RDV2)

These recovery measures are then used for:

- statistical comparison between groups  
- predictive modeling

---

# Dataset Structure

The repository contains processed outputs from the gait extraction pipeline.

Main folders:

```
BatchOutputs/
    controle/
    hemiparetique/
```

These folders contain the **extracted gait parameters for each subject**.


```

This file aggregates the extracted parameters for all subjects.

---

# GSRC Program Information

The file:

```
patients_cag.csv
```

contains information about whether a patient followed the **GSRC rehabilitation program**.

Example structure:

| patient_id | GSRC |
|------------|------|
| P01 | 1 |
| P02 | 0 |

Where:

- **1** = patient followed the GSRC program  
- **0** = patient did not follow the program  

This variable is later used as a **binary predictor in the machine learning models**.

---

# Source Code (src)

The core gait processing pipeline is implemented in the `src` folder.

```
src/
    data_loader.py
    gait_functions.py
    gait_processing.py
```

---

## data_loader.py

Handles dataset loading and preprocessing.

Main tasks:

- load sensor data from `.mat` or `.txt`
- reorder sensors to match anatomical positions
- load index files defining walking segments
- check data availability for each patient

Sensors are reordered into body positions:

```
0 waist
1 non-affected thigh
2 non-affected shank
3 non-affected foot
4 affected thigh
5 affected shank
6 affected foot
```

This ensures consistent biomechanical interpretation of the sensor signals.

---

## gait_functions.py

Contains the **core signal processing algorithms used for gait analysis**.

Main components include:

### Quaternion processing

- quaternion averaging  
- sign continuity correction  
- low-pass filtering  

### Sensor calibration

Calibration combines:

- gravity-based orientation estimation
- PCA-based estimation of the walking axis

### Joint angle computation

Joint angles are computed between adjacent body segments using quaternion rotations and Euler angle decomposition.

Estimated joint angles include:

- hip flexion
- knee flexion
- ankle flexion

---

## gait_processing.py

Implements the **complete gait processing pipeline**.

Main steps of the pipeline:

1. Data filtering  
2. Sensor calibration  
3. Joint angle computation  
4. Swing phase detection  
5. Step segmentation  
6. Extraction of gait parameters  

Extracted gait parameters include:

- stride length
- stride time
- stance time
- swing time
- stance duration
- walking speed

These parameters are computed separately for the **affected and non-affected limbs**.

---

# Longitudinal Dataset Construction

After gait extraction, a **longitudinal dataset** is constructed.

For each patient:

```
RDV2 parameters
RDV5 parameters
```

Recovery metrics are defined as:

```
Δparameter = RDV5 − RDV2
```

Examples include:

- ΔV → change in walking speed  
- ΔLP → change in step length  
- ΔFG → change in knee flexion peak  
- ΔFH → change in hip flexion  
- ΔEH → change in hip extension  

These variables represent **functional recovery indicators**.

---

# Machine Learning Model

The predictive modeling is implemented in:

```
train_models_results.ipynb
```

The objective is to predict recovery variables using baseline gait features.

Input features include:

- gait parameters measured at RDV2
- cadence
- step length
- walking speed
- GSRC participation indicator

The models evaluated include:

- Linear Regression
- Regularized regression models (Ridge / Lasso)

The GSRC variable is included as a **binary predictor** to evaluate its impact on recovery.

---

# Statistical Analysis

Two complementary approaches are used.

### 1. Direct Group Comparison

Patients are divided into two groups:

- GSRC
- non-GSRC

Differences in recovery metrics are evaluated using:

- Mann–Whitney tests
- effect size metrics

---

### 2. Regression Analysis

Regression models are used to estimate the contribution of GSRC while controlling for baseline values.

Example model:

```
ΔV ~ GSRC + V_RDV2
```

This allows evaluation of whether participation in the rehabilitation program significantly contributes to recovery.

---

# Key Outputs

The project produces the following outputs:

- extracted gait parameters for each subject
- longitudinal recovery metrics
- statistical comparisons between groups
- machine learning prediction results

These outputs allow investigation of how rehabilitation participation influences gait recovery trajectories.

---

# Repository Structure

```
stroke-gait-recovery-prediction/

src/
    data_loader.py
    gait_functions.py
    gait_processing.py

BatchOutputs/
    controle/
    hemiparetique/

QC_Outputs/

train_models_results.ipynb
script_QC.ipynb

patients_cag.csv

README.md
```

---

