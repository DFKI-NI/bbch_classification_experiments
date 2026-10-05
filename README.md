# BBCH Classification Experiments

This repository contains the official code and preprocessing pipelines for the MetroAgriFor-Paper Image-based BBCH-Stage Estimation of Maize Plants in the Leaf Development Phase using Multi-Task Learning.

It covers the preprocessing of the RocasMaize datasets, the training of multi-task models (MTL) that predict the BBCH stage of single maize plants,
and the evaluation of the one-stage approach (Mask2Former trained with 8 bbch classes) and the two-stage approach (Mask2Former trained with one maize class + MTL model)
presented in the paper.

# Setup

### 1. Clone the repository

```bash
git clone https://github.com/DFKI-NI/bbch_classification_experiments.git
cd bbch_classification_experiments
```

### 2. Download the datasets

Download all required files and store them in `datasets/` in the repository root according to the following structure:

- RocasMaize (images and annotations): RocasMaize doi soon
- RocasMaize_BBCH (BBCH annotations): https://doi.org/10.26249/FK2/OWSOUO
- Mask2Former predictions (maize_class and bbch_classes): https://cloud.dfki.de/owncloud/index.php/s/xfs9ndT33sxwqyz

```
datasets/
├── RocasMaize/
│   ├── annotations_datumaro/
│   │   └── all.json
│   └── images/
│       └── all/          # all full images (.png)
├── RocasMaize_BBCH/
│   └── annotations_datumaro/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── Predictions/
    ├── mask2former_maize_class_pred/
    │   └── annotations_fullres/
    │       └── test.json
    └── mask2former_bbch_classes_pred/
        └── annotations_fullres/
            └── test.json
```

RocasMaize_BBCH only contains annotations. All images are taken from RocasMaize.
The Mask2Former predictions are predictions for the used test set split in full image resolution and
are only needed for the evaluation (see [3. Evaluation](#3-evaluation)).

### 3. Container Environment

The repository contains a Dockerfile and a dev container configuration.

**VS Code:** Install the *Dev Containers* extension, open the repository folder and
run *Dev Containers: Reopen in Container*. VS Code builds the image and opens the
repository inside the container.

# Code execution
```
src/
├── data_preprocessing/   # builds all datasets based on RocasMaize and RocasMaize_BBCH
├── MTL/                  # multi-task model, training and evaluation
├── evaluation/           # one-stage and two-stage model pipeline evaluation
├── visualisations/       # confusion matrix and gradient conflict plots
├── metrics.py
└── utils.py
```

> **Note:** In general, commands should be run from the repository root. Some paths in the
> configs (e.g. `dataset_path`) are resolved from the current folder.

### 1. Preprocessing

```bash
python3 src/data_preprocessing/preprocess.py
```

This runs all preprocessing steps and creates
`datasets/RocasMaize_BBCH_processed/`:

```
datasets/RocasMaize_BBCH_processed/
├── mask2former/
│   ├── mask2former_maize_class/annotations/    # one class "maize" (BBCH 10-17)
│   └── mask2former_bbch_classes/annotations/   # one class per BBCH stage (10-17)
└── MTL/
    ├── images/{train,val,test}/                # one cropped image per maize plant
    ├── annotations/{train,val,test}.json       # labels and masks of each crop
    └── figures/                                # BBCH distribution and phenotype plots
```

The Mask2Former datasets only contain annotations. Their images are the full
images in `datasets/RocasMaize/images/all/`. The test sets in
`mask2former/` are the ground truth for the one-stage and two-stage evaluation.

The code and configurations from the RocasMaize main repository were used to train the Mask2Former models.


### 2. Training a MTL model

```bash
python3 src/MTL/main.py --config src/MTL/default_config.json
```

To choose a GPU, add `CUDA_VISIBLE_DEVICES=<id>` in front of the command. More
configs are in `src/MTL/experiment_configs/`. To reproduce our experiments, we provide the necessary config files in the `MTL/experiment_configs` directory.
For training, the config must contain `"mode": "train"`. Each training creates a new run folder containing the configs used for that run and the created model checkpoints.

To view the training curves:

```bash
mlflow ui --backend-store-uri sqlite:///training_runs/mlflow.db
```


### 3. Evaluation

To evaluate a trained run, set `"mode": "eval"` in the `config.json` of its run
folder. All evaluation results are logged to `training_runs/mlflow.db` and the confusion
matrices are saved in `training_runs/visualisations/`.

You can download the weights and final checkpoints of all trained models used in the paper here:
https://cloud.dfki.de/owncloud/index.php/s/xfs9ndT33sxwqyz


#### MTL model on the GT crops

```bash
python3 src/MTL/main.py --config training_runs/run_<timestamp>/config.json
```

Evaluates all tasks on the cropped test plants. The crops come from the GT masks,
so this does not include errors of the plant detection.

#### One-stage evaluation (Mask2Former bbch_classes)

```bash
python3 src/evaluation/one_stage_evaluation.py
```

Evaluates the BBCH classes predicted directly by Mask2Former. Options: `--pred`,
`--gt`, `--iou-threshold` (default 0.5) and `--no-plot`. MLflow experiment:
`one_stage_evaluation`.

#### Two-stage evaluation (Mask2Former maize_class + MTL)

```bash
python3 src/evaluation/two_stage_evaluation.py --config training_runs/run_<timestamp>/config.json
```

Cuts the plants predicted by Mask2Former out of the full images and predicts their
BBCH stage with the MTL model. Add `--use_baseline_masks` to use the masks of the
bbch_classes model instead of the maize_class model. MLflow experiment:
`two_stage_evaluation`.

- On the first run, the predicted masks are cut out and stored in
  `datasets/Predictions/<model>_pred/cut_images/`.
- The predictions are saved in
  `datasets/Predictions/two_stage_<run>_<checkpoint>/`.

Both evaluations print the metrics per BBCH class for all plants and separately for
plants with and without visible damage, followed by the localization metrics.

#### Gradient conflict matrix

```bash
python3 src/visualisations/gradient_matrix.py --config training_runs/run_<timestamp>/config.json
```

Plots the gradient similarities between all task pairs of a run. Only works for
runs with all 7 tasks. The plot is saved in `training_runs/visualisations/`.


# Citation

If you find this work useful in your research, please cite:
```

```


# References

Our code is partially based on the following code bases:
* [LibMTL](https://github.com/median-research-group/libmtl)
* [PyTorch](https://github.com/pytorch/pytorch)
* [Transformers](https://github.com/huggingface/transformers)
