# Uncertainty-Aware Retinal Disease Classification Under Domain Shift in Optical Coherence Tomography

This repository contains code for the project:

**Uncertainty-Aware Retinal Disease Classification Under Domain Shift in Optical Coherence Tomography**

The project investigates whether an Evidential Neural Network (ENN) can improve the reliability of retinal optical coherence tomography (OCT) disease classification compared with a standard ResNet18 softmax baseline, especially when evaluated on external datasets collected under different scanner and acquisition conditions.

## Project Overview

Deep learning models for retinal OCT classification often perform well on internal benchmark datasets, but their reliability may decrease when applied to external data from different sources. This project studies this problem by comparing:

1. A standard **ResNet18 softmax classifier**
2. An **Evidential Neural Network (ENN)** using the same ResNet18 backbone

The ENN predicts Dirichlet evidence rather than only softmax probabilities, allowing estimation of both class probabilities and predictive uncertainty. The main goal is not only to maximize internal accuracy, but also to evaluate robustness, calibration, and uncertainty behavior under domain shift.

## Datasets

### Development Dataset

The model was trained and internally evaluated using the Kermany OCT dataset:

* Dataset: Kermany OCT
* Link: [https://data.mendeley.com/datasets/rscbjbr9sj/1](https://data.mendeley.com/datasets/rscbjbr9sj/1)
* Classes: `CNV`, `DME`, `DRUSEN`, `NORMAL`
* Training set: 20,000 images, with 5,000 images per class
* Internal test set: 1,000 images, with 250 images per class

### External Datasets

External evaluation was performed on three independent OCT datasets:

| Dataset           | Link                                                                                                                                     |        Classes Used | Number of Images |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------: | ---------------: |
| Srinivasan / DUKE | [https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm](https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm)           | DRUSEN, DME, NORMAL |            3,231 |
| OCTDL             | [https://data.mendeley.com/datasets/sncdhf53xc/4](https://data.mendeley.com/datasets/sncdhf53xc/4)                                       |    CNV, DME, NORMAL |            2,064 |
| OCTID             | [https://www.openicpsr.org/openicpsr/project/108503/version/V1/view](https://www.openicpsr.org/openicpsr/project/108503/version/V1/view) |         CNV, NORMAL |              261 |

These external datasets differ in scanner type, acquisition protocol, and class composition, allowing evaluation of cross-dataset generalization under domain shift.

## Important Data Note

Raw OCT images are **not included** in this repository due to dataset size and licensing restrictions.

Users should download the datasets from their original sources and organize them locally according to the expected folder or CSV structure. Metadata CSV files may be provided to show the expected format, but image paths must be updated for each local machine or server.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── configs/
│   ├── baseline_resnet18.yaml
│   ├── enn_lam0p7.yaml
│   ├── tau_threshold.yaml
│   └── selective_prediction.yaml
│
├── metadata/
│   ├── dataset_description.md
│   ├── kermany/
│   │   ├── train_subset.csv
│   │   ├── val.csv
│   │   └── test_official.csv
│   └── external/
│       ├── srinivasan_labels.csv
│       ├── octdl_labels.csv
│       └── octid_labels.csv
│
├── src/
│   ├── training/
│   │   ├── train_baseline_kermany.py
│   │   └── train_enn_kermany.py
│   │
│   ├── evaluation/
│   │   ├── eval_external_baseline.py
│   │   ├── eval_external_enn.py
│   │   ├── tau_sweep_normal_threshold.py
│   │   ├── evaluate_umass_filtering.py
│   │   └── selective_coverage_accuracy_umass.py
│   │
│   ├── visualization/
│   │   ├── plot_main_results.py
│   │   ├── plot_reliability.py
│   │   └── gradcam.py
│   │
│   └── data/
│       ├── datasets.py
│       └── transforms.py
│
├── scripts/
│   ├── 01_train_baseline.sh
│   ├── 02_train_enn.sh
│   ├── 03_eval_external_baseline.sh
│   ├── 04_eval_external_enn.sh
│   ├── 05_tau_sweep.sh
│   ├── 06_umass_filtering.sh
│   └── 07_selective_coverage.sh
│
└── results/
    └── example_outputs/
```

Depending on the current cleanup stage of the repository, some folders or filenames may differ slightly. The intended final structure above separates training, evaluation, visualization, metadata, and reproducibility scripts.

## Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` should include:

```text
torch
torchvision
numpy
pandas
pillow
scikit-learn
matplotlib
tqdm
pyyaml
```

## Expected CSV Format

Training and evaluation scripts expect CSV files containing image paths and labels.

A typical CSV should contain columns such as:

```text
dst_path,label,split,patient_id
```

Example:

```text
/path/to/image1.jpeg,CNV,train,patient_001
/path/to/image2.jpeg,NORMAL,test,patient_002
```

For external datasets, the CSV should contain at least:

```text
dst_path,label
```

or:

```text
relpath,label
```

where `relpath` can be combined with a provided `data_root`.

## Model Training

### 1. Train the Softmax Baseline

The baseline model is a ResNet18 classifier trained using cross-entropy loss.

Example command:

```bash
python src/training/train_baseline_kermany.py \
  --arch resnet18 \
  --pretrained 1 \
  --img_size 224 \
  --batch_size 64 \
  --epochs 50 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --balance fixed_per_class \
  --per_class 5000 \
  --scheduler cosine \
  --early_stop 0
```

Expected outputs include:

```text
best.pt
history.json
test_metrics.json
test_uncertainty.csv
```

### 2. Train the Evidential Neural Network

The ENN uses the same ResNet18 backbone but replaces the softmax output with an evidential output layer.

The model outputs non-negative evidence values. These are converted to Dirichlet parameters:

```text
alpha = evidence + 1
```

The expected class probability is computed from the Dirichlet mean, and predictive uncertainty is estimated using uncertainty mass:

```text
u_mass = K / S
```

where `K` is the number of classes and `S` is the total Dirichlet strength.

Example command:

```bash
python src/training/train_enn_kermany.py \
  --train_csv metadata/kermany/train_subset.csv \
  --val_csv metadata/kermany/val.csv \
  --test_csv metadata/kermany/test_official.csv \
  --arch resnet18 \
  --pretrained 1 \
  --img_size 224 \
  --batch_size 64 \
  --epochs 50 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --lam 0.7 \
  --early_stop 0 \
  --runs_dir runs/enn_kermany
```

Expected outputs include:

```text
best.pt
history.json
test_metrics.json
test_uncertainty_with_probs.csv
```

## External Evaluation

### Baseline External Evaluation

Example command:

```bash
python src/evaluation/eval_external_baseline.py \
  --ckpt runs/baseline_kermany/best.pt \
  --data_root data/octdl \
  --labels_csv metadata/external/octdl_labels.csv \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 64 \
  --use_split all \
  --out_json results/octdl_baseline_external.json \
  --out_csv results/octdl_baseline_predictions.csv
```

### ENN External Evaluation

Example command:

```bash
python src/evaluation/eval_external_enn.py \
  --ckpt runs/enn_kermany/best.pt \
  --data_root data/octdl \
  --labels_csv metadata/external/octdl_labels.csv \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 64 \
  --use_split all \
  --out_json results/octdl_enn_external.json \
  --out_csv results/octdl_enn_predictions.csv
```

The ENN prediction CSV should include columns such as:

```text
path
y_true
y_pred
y_true_label
y_pred_label
prob_CNV
prob_DME
prob_DRUSEN
prob_NORMAL
alpha_CNV
alpha_DME
alpha_DRUSEN
alpha_NORMAL
u_mass
p_max
correct
```

These columns are used by the threshold-tuning and selective-prediction scripts.

## Threshold Tuning

A NORMAL-class threshold was evaluated to reduce disease-to-normal errors.

The rule is:

```text
if prob_NORMAL >= tau:
    predict NORMAL
else:
    predict the disease class with the highest probability
```

Example command:

```bash
python src/evaluation/tau_sweep_normal_threshold.py \
  --internal_csv results/internal_enn_predictions.csv \
  --srin_csv results/srinivasan_enn_predictions.csv \
  --octdl_csv results/octdl_enn_predictions.csv \
  --octid_csv results/octid_enn_predictions.csv \
  --tau_csv configs/tau_values.csv \
  --out_csv results/tau_sweep_summary.csv \
  --criterion safety
```

The final threshold used in the paper was:

```text
tau = 0.27
```

## Uncertainty-Based Filtering

Uncertainty-based filtering uses ENN uncertainty mass, `u_mass`, to retain lower-uncertainty cases.

For a fixed threshold such as:

```text
u_mass <= 0.85
```

the script keeps only predictions with uncertainty mass below or equal to the threshold and computes performance on retained cases.

Example command:

```bash
python src/evaluation/evaluate_umass_filtering.py \
  --internal_csv results/internal_enn_predictions.csv \
  --srin_csv results/srinivasan_enn_predictions.csv \
  --octdl_csv results/octdl_enn_predictions.csv \
  --octid_csv results/octid_enn_predictions.csv \
  --umass_threshold 0.85 \
  --out_csv results/umass_filtering_summary.csv
```

## Selective Prediction

Selective prediction evaluates model performance after retaining only the lowest-uncertainty cases.

The procedure is:

```text
1. Sort cases by u_mass from low to high.
2. Retain the lowest-uncertainty cases at a selected coverage level.
3. Compute accuracy and other metrics only on retained cases.
```

Example command:

```bash
python src/evaluation/selective_coverage_accuracy_umass.py \
  --internal_csv results/internal_enn_predictions.csv \
  --srin_csv results/srinivasan_enn_predictions.csv \
  --octdl_csv results/octdl_enn_predictions.csv \
  --octid_csv results/octid_enn_predictions.csv \
  --out_csv results/selective_coverage_accuracy_umass.csv
```

This produces a CSV table showing retained-case accuracy from 100% coverage down to lower coverage levels, such as 90%, 80%, 70%, 60%, and 50%.

## Evaluation Metrics

The main evaluation metrics are:

* Multiclass accuracy
* Disease-versus-normal ROC-AUC
* Macro false-negative rate
* Macro false-positive rate
* Expected calibration error, or ECE
* Brier score
* Retained coverage for uncertainty filtering and selective prediction

External results are reported for each dataset and also averaged across the three external datasets.

## Main Findings

The softmax baseline achieved higher internal accuracy, but showed a larger internal-to-external accuracy drop under domain shift.

The ENN showed more stable external performance and provided useful uncertainty estimates. Threshold tuning and uncertainty-based selective prediction further improved retained-case performance and reduced false-negative and false-positive rates.

At approximately 90% retained coverage, selective prediction improved average external accuracy and reduced error rates on retained cases, supporting the use of ENN uncertainty as a decision-support signal.

## Reproducibility Notes

To reproduce the experiments:

1. Download the datasets from the original sources.
2. Update the CSV files so that `dst_path` points to the correct local image paths.
3. Install dependencies using `requirements.txt`.
4. Train the baseline and ENN models.
5. Run external evaluation on Srinivasan, OCTDL, and OCTID.
6. Run threshold tuning and selective prediction scripts.
7. Generate summary tables and figures.

Because raw OCT datasets are not redistributed in this repository, exact reproduction requires access to the original datasets and correct local path configuration.

## Citation

If you use this code, please cite the corresponding paper:

```text
Chen, A. X., Jafarpisheh, N., Namdar, K., & Tyrrell, P. N.
Uncertainty-Aware Retinal Disease Classification Under Domain Shift in Optical Coherence Tomography.
```

## Contact

For questions about the repository, please contact:

```text
Abigail Xi Chen
```

## License

This repository is released for academic and research use. Please see the `LICENSE` file for details.
