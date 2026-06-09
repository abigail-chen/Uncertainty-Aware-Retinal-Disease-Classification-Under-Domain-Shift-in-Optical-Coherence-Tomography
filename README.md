# Uncertainty-Aware Retinal Disease Classification Under Domain Shift in Optical Coherence Tomography

This repository contains code for the paper:

**Uncertainty-Aware Retinal Disease Classification Under Domain Shift in Optical Coherence Tomography**

The project investigates whether an Evidential Neural Network (ENN) can improve the reliability of retinal optical coherence tomography (OCT) disease classification compared with a standard ResNet18 softmax baseline, especially under domain shift across external datasets.

## Overview

Deep learning models for retinal OCT classification can perform well on internal benchmark datasets but may become less reliable when evaluated on external data collected with different scanners, acquisition protocols, or class distributions.

This project compares two models:

1. **Softmax baseline:** ResNet18 with a standard softmax classifier.
2. **Uncertainty-aware model:** Evidential Neural Network using the same ResNet18 backbone.

The ENN produces Dirichlet-based evidence, allowing prediction of both class probabilities and uncertainty. The main goal is to evaluate robustness, calibration, and uncertainty-aware decision support under domain shift.

## Datasets

### Development Dataset

The Kermany OCT dataset was used for model training and internal testing.

| Dataset     | Official source                                 | Classes used             | Use                           |
| ----------- | ----------------------------------------------- | ------------------------ | ----------------------------- |
| Kermany OCT | https://data.mendeley.com/datasets/rscbjbr9sj/1 | CNV, DME, DRUSEN, NORMAL | Training and internal testing |

The final experiments used:

* 20,000 training images, with 5,000 images per class
* 1,000 internal test images, with 250 images per class

### External Evaluation Datasets

External evaluation was performed on three independent OCT datasets:

| Dataset           | Official source                                                    | Classes used        | Number of images |
| ----------------- | ------------------------------------------------------------------ | ------------------- | ---------------: |
| Srinivasan / DUKE | https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm      | DRUSEN, DME, NORMAL |            3,231 |
| OCTDL             | https://data.mendeley.com/datasets/sncdhf53xc/4                    | CNV, DME, NORMAL    |            1,710 |
| OCTID             | https://www.openicpsr.org/openicpsr/project/108503/version/V1/view | CNV, NORMAL         |              261 |

These datasets differ in class composition, scanner type, and acquisition conditions, providing a realistic evaluation of model behavior under domain shift.


## Data Availability

Raw OCT images are **not included** in this repository because of dataset size and redistribution restrictions.

Users should download the original datasets from their official sources and update the CSV files so that image paths point to the correct local locations.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── 01_train_baseline.sh
│   ├── 02_train_enn.sh
│   ├── 03_eval_external_baseline.sh
│   ├── 04_eval_external_enn.sh
│   ├── 05_tau_sweep_normal_threshold.sh
│   ├── 06_umass_thresholding.sh
│   └── 07_selective_coverage_accuracy_umass.sh
│
└── src/
    ├── data/
    │   ├── Kermany/
    │   │   ├── train_subset.csv
    │   │   ├── val.csv
    │   │   └── test_official.csv
    │   │
    │   ├── external_OCTDL/
    │   │   └── labels_for_eval_4class_MAPPED.csv
    │   │
    │   ├── external_OCTID/
    │   │   └── labels_for_eval_4class_MAPPED.csv
    │   │
    │   └── external_Srinivasan/
    │       └── labels_for_eval_4class_MAPPED.csv
    │
    ├── training/
    │   ├── train_baseline_Kermany.py
    │   └── train_enn_kermany.py
    │
    └── evaluation/
        ├── eval_external_baseline.py
        ├── eval_external_enn.py
        ├── tau_sweep_normal_threshold.py
        ├── umass_thresholding.py
        └── selective_coverage_accuracy_umass.py
```

The `src/training/` folder contains the main training code for the softmax baseline and ENN. The `src/evaluation/` folder contains external evaluation, NORMAL-threshold tuning, u-mass thresholding, and selective prediction analysis. The `scripts/` folder provides reproducible command-line wrappers for running the training and evaluation workflow.


## Environment Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include PyTorch, torchvision, NumPy, pandas, Pillow, scikit-learn, and matplotlib.

## Expected CSV Format

The training and evaluation scripts use CSV files to locate images and labels.

For Kermany training and internal testing, the expected columns include:

```text
dst_path,label,patient_id,split
```

For external evaluation, the expected columns include at least:

```text
dst_path,label,patient_id
```

The `dst_path` column should contain the full path to each OCT image on the local machine or server.

## Model Training

### Baseline Model

The baseline is a ResNet18 softmax classifier trained using cross-entropy loss.

The final experiment used:

* ImageNet-pretrained ResNet18
* Image size: 224 × 224
* Batch size: 64
* Epochs: 50
* Optimizer: AdamW
* Learning rate: 3e-4
* Weight decay: 1e-5
* Random seed: 42
* Early stopping disabled

Training is handled by:

```text
src/training/train_baseline_Kermany.py
```

### Evidential Neural Network

The ENN uses the same ResNet18 backbone but replaces the standard softmax output with an evidential output layer. The model estimates class probabilities and predictive uncertainty through Dirichlet evidence.

The final ENN setting used:

* ResNet18 backbone
* Batch size: 64
* Epochs: 50
* Learning rate: 3e-4
* Weight decay: 1e-5
* λ = 0.7
* KL warm-up during the first 10 epochs
* Early stopping disabled

Training is handled by:

```text
src/training/train_enn_kermany.py
```

The main output files include:

```text
best.pt
history.json
test_metrics.json
test_uncertainty_with_probs.csv
```

## External Evaluation

External evaluation compares model performance on OCTDL, OCTID, and Srinivasan.

Baseline external evaluation is handled by:

```text
src/evaluation/eval_external_baseline.py
```

ENN external evaluation is handled by:

```text
src/evaluation/eval_external_enn.py
```

The ENN prediction outputs are used by the threshold-tuning and uncertainty-based evaluation scripts.

## Threshold Tuning

NORMAL-threshold tuning was used to reduce disease-to-normal misclassification.

The rule is:

```text
if prob_NORMAL >= tau:
    predict NORMAL
else:
    predict the disease class with the highest probability
```

The final threshold used in the paper was:

```text
tau = 0.27
```

The threshold sweep is handled by:

```text
src/evaluation/tau_sweep_normal_threshold.py
```

## Uncertainty-Based Filtering

The ENN produces an uncertainty mass value, `u_mass`, where larger values indicate higher uncertainty.

Uncertainty-based filtering was evaluated using:

```text
u_mass <= 0.85
```

This keeps lower-uncertainty predictions and evaluates performance only on retained cases.

The corresponding script is:

```text
src/evaluation/umass_thresholding.py
```

## Selective Prediction

Selective prediction evaluates how performance changes as the most uncertain cases are deferred.

The procedure is:

```text
1. Sort predictions by u_mass from low to high.
2. Retain the lowest-uncertainty cases.
3. Compute accuracy on retained cases.
4. Repeat across different coverage levels.
```

Selective coverage analysis is handled by:

```text
src/evaluation/selective_coverage_accuracy_umass.py
```

In the final paper, selective prediction was reported at approximately 90% retained coverage.

## Evaluation Metrics

The main metrics are:

* Multiclass accuracy
* Disease-versus-normal ROC-AUC
* Macro false-negative rate
* Macro false-positive rate
* Expected calibration error
* Adaptive expected calibration error
* Brier score
* Retained coverage for selective prediction

External metrics are reported for each dataset and also averaged across the three external datasets.

## Main Findings

The softmax baseline achieved higher internal accuracy, but showed a larger internal-to-external accuracy drop under domain shift.

The ENN did not maximize in-distribution accuracy, but it provided more stable external performance and more informative uncertainty estimates. Threshold tuning and uncertainty-based selective prediction further improved retained-case performance and reduced both false-negative and false-positive rates.

At approximately 90% retained coverage, selective prediction improved average external accuracy and reduced error rates on retained cases, supporting uncertainty as a practical decision-support signal for retinal OCT classification under domain shift.

## Reproducibility Notes

To reproduce the workflow:

1. Download the original OCT datasets.
2. Update the CSV files so that `dst_path` points to local image paths.
3. Install dependencies using `requirements.txt`.
4. Train the baseline and ENN models.
5. Evaluate both models on the external datasets.
6. Run threshold tuning and uncertainty-based selective prediction analysis.

Because raw OCT images are not redistributed in this repository, exact reproduction requires access to the original datasets and correct local path configuration.


## Contact

For questions, please contact:

```text
abigail.chen@mail.utoronto.ca
```

## License

MIT License
