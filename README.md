# Predictive Modelling of Readout Error Drift in Quantum Processors Using Machine Learning

**A machine learning study of temporal readout-error behaviour in superconducting quantum processors using IBM Quantum calibration data.**


\


## Overview

Quantum processors are sensitive to noise and environmental changes. One important source of error is **imperfect qubit readout**: when a quantum state is measured, the classical result may not correctly represent the state of the qubit.

Readout characteristics can change over time. This behaviour, commonly described as *readout-error drift*, makes calibration monitoring important for maintaining reliable quantum computation.

This project investigates whether historical calibration measurements and temporal features can be used to predict the **next recorded readout-error observation** for individual backend–qubit series.

We use classical machine learning models to analyse temporal patterns in quantum processor calibration data, compare their predictive performance, and investigate the potential of data-driven methods for calibration monitoring.

> **Important scope:** This project applies classical machine learning to quantum-computing calibration data. It does not train a quantum neural network or directly simulate quantum circuits.

## Research Objective

The primary objective is to develop and evaluate predictive models for readout-error behaviour using historical observations.

The project focuses on four questions:

1. Can historical readout-error measurements help predict the next recorded observation?
2. Do temporal features such as lag values, rolling statistics, and calibration age provide useful predictive information?
3. How do different machine learning approaches compare on this prediction task?
4. Can the resulting models support more informed monitoring of readout-error behaviour?

## Dataset

The project uses the **Qiskit Calibration Drift** dataset.

- **Source:** [phanerozoic/qiskit-calibration-drift](https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift)
- **Domain:** Quantum processor calibration measurements
- **Prediction variable:** `readout_error`
- **Series definition:** Individual combinations of `backend` and `qubit_a`
- **Backends represented in the analysed data:** `ibm_fez`, `ibm_kingston`, `ibm_marrakesh`, and `ibm_torino`

The dataset contains time-associated calibration measurements. The observations are organised chronologically within each backend–qubit series before temporal features and prediction targets are constructed.

The dataset is not bundled in this repository by default. Please obtain it from the original source and follow its licensing and usage conditions.

## Methodology

The workflow consists of data preparation, temporal feature engineering, chronological data splitting, model training, and performance evaluation.

### 1. Data preparation

The data preparation workflow includes:

- Loading and inspecting the calibration measurements.
- Parsing observation timestamps.
- Organising observations by backend and qubit.
- Sorting measurements chronologically.
- Preparing the readout-error series for temporal analysis.
- Handling invalid or unavailable feature values.

### 2. Temporal feature engineering

Features are constructed independently within each backend–qubit series.

| Feature group           | Features                                                          | Purpose                                     |
| ----------------------- | ----------------------------------------------------------------- | ------------------------------------------- |
| Current measurement     | `current_error`                                                   | Most recent observed readout error          |
| Lag features            | `lag_1`, `lag_2`, `lag_3`, `lag_5`                                | Historical error measurements               |
| Rolling statistics      | Rolling mean and standard deviation over 3, 5 and 10 observations | Recent level and variability                |
| Temporal information    | `time_gap_hours`                                                  | Time elapsed since the previous observation |
| Change information      | `error_change`, `error_change_rate`                               | Magnitude and rate of recent change         |
| Calibration information | `calibration_age_seconds`                                         | Calibration-age information                 |
| Categorical information | `backend`, `qubit_a`                                              | Backend and qubit identity                  |

The model uses **17 input features: 15 numerical features and 2 categorical features**.

Rolling statistics are calculated using the current observation and preceding observations in the same series. The next observation is kept separate as the prediction target.

### 3. Prediction target

The target is:

`target_next_error`

For each observation, this represents the next recorded `readout_error` value within the same backend–qubit series.

This is a **next-observation prediction task**, not a fixed-horizon forecast. The time interval between the current and next observation can vary.

The associated `target_gap_hours` is retained as evaluation metadata and is not used as a model input.

### 4. Chronological data splitting

To respect the temporal structure of the data, observations are not randomly shuffled.

Each backend–qubit series is split independently:

- 70% training
- 15% validation
- 15% testing

The split preserves chronological order within each series. This evaluates predictions on later observations from series represented in the training data; it does not establish generalisation to entirely unseen backends or qubits.

### 5. Model development

The experiments compare multiple approaches:

- Persistence baseline
- Linear Regression
- Random Forest Regressor
- Tuned Random Forest
- LightGBM Regressor
- Tuned LightGBM
- LSTM

The models represent different approaches to learning from historical measurements, ranging from simple baselines and linear relationships to ensemble learning and sequence modelling.

Hyperparameter tuning was also performed for selected models.

## Validation Results

The following table reports the validation leaderboard from the documented model-comparison run.

| Model                  |          MAE |         RMSE |           R² |
| ---------------------- | -----------: | -----------: | -----------: |
| Tuned Random Forest    | **0.008712** | **0.026254** | **0.811788** |
| Linear Regression      |     0.008743 |     0.026294 |     0.811217 |
| Original Random Forest |     0.008733 |     0.026322 |     0.810805 |
| LSTM                   |     0.008710 |     0.026404 |     0.809632 |
| Tuned LightGBM         |     0.008976 |     0.027528 |     0.793083 |
| Original LightGBM      |     0.009064 |     0.028134 |     0.783867 |
| Persistence baseline   |     0.013202 |     0.043243 |     0.654033 |

**Interpretation**

- Tuned Random Forest achieved the lowest validation RMSE and highest validation R² in this comparison.
- LSTM achieved a marginally lower MAE than the other models, although its RMSE and R² were slightly weaker than Tuned Random Forest.
- Both Random Forest and Linear Regression performed similarly in this run.
- The persistence baseline provides a reference point for assessing the benefit of learned models.
- Tuned LightGBM was selected as the team's primary integration model, with Tuned Random Forest retained as an alternative. This integration choice should not be confused with being the highest-scoring validation model.

These values are validation results from the documented comparison run, not a claim of universal model performance. Results from separate experiments should only be compared when their datasets, feature definitions, splits, and evaluation procedures are consistent.

## Evaluation Metrics

Three regression metrics are used:

- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual readout-error values.
- **Root Mean Squared Error (RMSE):** Penalises larger prediction errors more heavily.
- **R² score:** Measures how much better the model performs relative to a mean-prediction reference on the evaluated data.

Because `readout_error` is represented as a fraction, these errors are expressed in the same fractional units. R² is not classification accuracy.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- LightGBM
- TensorFlow / Keras (for LSTM experiments)
- Jupyter Notebook / Google Colab
- Hugging Face Datasets

## Repository Structure

The repository is intended to follow a structure similar to:

```text
quantum-readout-error-prediction/
│
├── notebooks/
│   ├── yuvraj_model_experiments.ipynb
│   └── collaborator_model_experiments.ipynb
│
├── models/
│   ├── tuned_lightgbm_model.joblib
│   └── tuned_random_forest_model.joblib
│
├── results/
│   ├── validation_comparison.csv
│   └── figures/
│
├── docs/
│   ├── methodology.md
│   └── feature_engineering.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

Files may be added as they are prepared. Large datasets and model binaries can be distributed separately if repository size or hosting limits make that more appropriate.

## Reproducibility

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   cd YOUR-REPOSITORY
   ```

2. Create and activate a Python environment:

   ```bash
   python -m venv .venv
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Obtain the dataset from the Hugging Face source linked above.

5. Open the experiment notebook in Jupyter or Google Colab and update the dataset path to match your environment.

6. Execute the notebook cells in order to reproduce the data preparation, feature engineering, training, and evaluation workflow.

Exact package versions and dataset preparation steps should be recorded alongside the final notebook to support reproducibility.

## Limitations and Future Work

This project is an initial predictive study, and several directions remain open for further investigation:

- **Variable prediction horizon:** The next recorded observation may occur after different time intervals. Future work could evaluate fixed-horizon forecasts.
- **Unseen hardware generalisation:** The per-series split evaluates future observations from known backend–qubit series. Separate experiments are needed to evaluate unseen qubits and backends.
- **Changing calibration behaviour:** Hardware conditions and drift patterns may change over time, potentially reducing performance under new operating conditions.
- **Uncertainty estimation:** Prediction intervals could help communicate confidence rather than providing only point predictions.
- **Model interpretability:** Feature-importance and explainability analyses could provide more insight into which temporal and calibration variables contribute to predictions.
- **Operational evaluation:** Further work could investigate whether predictions can support calibration scheduling or monitoring decisions.

## Project Significance

The project explores a practical intersection of quantum computing, data science, and machine learning.

Rather than treating calibration measurements as isolated values, it studies their temporal structure and asks whether historical behaviour contains useful information about subsequent readout-error observations.

The broader motivation is to investigate how data-driven predictive methods could contribute to the monitoring and understanding of noisy quantum hardware.

## Contributors

Developed as a collaborative machine learning research project.

- **Yuvraj  Tayal:** Model development, experiments, feature engineering and model evaluation.
- **Project collaborators:** Dataset preparation, additional model experiments and application integration.

Individual contributions and experiment ownership should be documented alongside the corresponding notebooks.

## Acknowledgements

- The contributors to the Qiskit Calibration Drift dataset.
- The Qiskit and IBM Quantum ecosystems for making quantum-computing calibration research accessible.
- The open-source Python machine learning ecosystem.

---

**Research area:** Quantum Computing · Machine Learning · Time-Series Prediction · Quantum Hardware Calibration
