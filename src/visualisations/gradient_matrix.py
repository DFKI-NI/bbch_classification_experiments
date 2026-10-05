"""Plot the gradient conflict matrices of one MTL run with all 7 tasks.
 
During training, the cosine and magnitude similarity of the task gradients are
logged to MLflow for each pair of tasks (see MLflowPerformanceMeter in utils.py).
This script loads these values of one run, averages them over all epochs and
plots two 7 x 7 matrices side by side:
 
- Angle-based conflict: mean cosine similarity (-1 to 1). Values below 0 mean the
  gradients of both tasks point in opposite directions.
- Magnitude conflict: mean magnitude similarity (0 to 1). Values close to 0 mean
  the gradients of both tasks have very different sizes.
 
The run is read from the config: run_dir (with mlflow.db), experiment_name and
evaluate_run. If evaluate_run is not found, the newest run of the experiment is
used. The script stops if the run does not contain all 7 tasks. The plot is saved
as {run_name}_gradient_matrix.png in run_dir/visualisations.
 
Usage (run_dir is the folder of a trained MTL run):
    python3 src/visualisations/gradient_matrix.py --config <run_dir>/config.json
"""


import argparse
import json
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ALL_TASKS = [
    'bbch_classification', 
    'bbch_regression', 
    'pprcm_regression', 
    'damaged_classification', 
    'semantic_segmentation', 
    'distance_regression', 
    'direction_regression'
]


def parse_args():
    """Parse the command line arguments.
 
    Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Plot 7x7 Gradient Conflict Matrix for a single MTL run from MLflow.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to config.json"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load a JSON config file.
 
    Returns the config as a dict.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_display_names(tasks):
    """Convert task names into readable labels for the plot.
 
    For example "bbch_regression" becomes "BBCH Regression" and "pprcm_regression"
    becomes "PPR Regression".
 
    Returns a list with the labels.
    """
    display_names = []
    for t in tasks:
        name = t.replace('_', ' ').title()
        name = name.replace('Bbch', 'BBCH').replace('Pprcm', 'PPR')
        display_names.append(name)
    return display_names


def load_run_metrics_as_df(client, run_id):
    """Load the cosine and magnitude similarities of a run from MLflow.
 
    Each similarity metric becomes one column, named cos_{task1}_vs_{task2} or
    mag_{task1}_vs_{task2}, with one row per logged epoch.
 
    Returns the values as a DataFrame (empty if the run has no similarities).
    """
    run = client.get_run(run_id)
    metric_keys = run.data.metrics.keys()
    
    data_dict = {}
    for key in metric_keys:
        if key.startswith("Similarities_Cosine/") or key.startswith("Similarities_Magnitude/"):
            history = client.get_metric_history(run_id, key)
            steps = [m.step for m in history]
            vals = [m.value for m in history]
            
            prefix = "cos_" if "Cosine" in key else "mag_"
            pair_name = key.split("/")[-1]
            col_name = f"{prefix}{pair_name}"
            
            if 'step' not in data_dict:
                data_dict['step'] = steps
            data_dict[col_name] = vals

    return pd.DataFrame(data_dict)


def validate_all_tasks_present(df, tasks):
    """Check that each task appears in at least one similarity column.
 
    Prints the missing tasks if some are missing.
 
    Returns True if all tasks are present, otherwise False.
    """
    found_tasks = set()
    
    for task in tasks:
        for col in df.columns:
            if task in col:
                found_tasks.add(task)
                break

    missing_tasks = set(tasks) - found_tasks
    if missing_tasks:
        print(f"\n[Error] Only {len(found_tasks)} of {len(tasks)} tasks are present.")
        print(f"Missing Tasks ({len(missing_tasks)}): {list(missing_tasks)}")
        return False
    
    return True


def plot_gradient_matrix(df, tasks, save_path, run_name=""):
    """Plot the cosine and magnitude matrix of all task pairs and save the plot.
 
    Each cell is the mean of the similarity over all epochs. Only the lower triangle
    is shown, because both matrices are symmetric. The diagonal is 1.
    """
    num_tasks = len(tasks)
    display_names = get_display_names(tasks)

    cos_matrix, mag_matrix = np.ones((num_tasks, num_tasks)), np.ones((num_tasks, num_tasks))

    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            t1, t2 = tasks[i], tasks[j]
            c_col, m_col = f'cos_{t1}_vs_{t2}', f'mag_{t1}_vs_{t2}'
            c_col_alt, m_col_alt = f'cos_{t2}_vs_{t1}', f'mag_{t2}_vs_{t1}'
            
            c_key = c_col if c_col in df.columns else (c_col_alt if c_col_alt in df.columns else None)
            m_key = m_col if m_col in df.columns else (m_col_alt if m_col_alt in df.columns else None)

            if c_key and m_key:
                cv, mv = df[c_key].mean(), df[m_key].mean()
                cos_matrix[i, j] = cos_matrix[j, i] = cv
                mag_matrix[i, j] = mag_matrix[j, i] = mv

    mask = np.triu(np.ones_like(cos_matrix, dtype=bool), k=1)
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))

    ticks_13 = np.linspace(-1, 1, 11)
    ticks_2 = np.linspace(0, 1, 11)

    sns.heatmap(cos_matrix, mask=mask, annot=True, fmt=".2f", ax=ax[0], square=True,
                cmap='RdBu', center=0, vmin=-1, vmax=1, 
                cbar_kws={'shrink': .7, 'ticks': ticks_13},
                xticklabels=display_names, yticklabels=display_names)
    ax[0].set_title('Angle-Based Gradient Conflict', fontweight='bold', pad=20, fontsize=14)

    sns.heatmap(mag_matrix, mask=mask, annot=True, fmt=".2f", ax=ax[1], square=True,
                cmap='RdBu', center=0.5, vmin=0, vmax=1, 
                cbar_kws={'shrink': .7, 'ticks': ticks_2},
                xticklabels=display_names, yticklabels=display_names)
    ax[1].set_title('Magnitude Gradient Conflict', fontweight='bold', pad=20, fontsize=14)

    for a in ax:
        a.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        a.set_yticklabels(display_names, rotation=0, fontsize=10)
        a.tick_params(axis='both', which='both', length=0)
        a.set_xlabel('')
        a.set_ylabel('')

    plt.suptitle(f"Gradient Similarities - {run_name}", fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.85, wspace=0.35)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gradient matrix saved at: {save_path}")


def main():
    """Find the run in MLflow, load its similarities and plot the matrices.
 
    Exits with code 1 if the run does not contain all 7 tasks.
    """
    args = parse_args()
    config = load_config(args.config)

    run_dir = config.get("run_dir", "training_runs")
    experiment_name = config.get("experiment_name")
    target_run_name = config.get("evaluate_run")

    db_path = os.path.abspath(os.path.join(run_dir, "mlflow.db"))
    tracking_uri = f"sqlite:///{db_path}"

    output_dir = os.path.join(run_dir, "visualisations")
    os.makedirs(output_dir, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: Experiment '{experiment_name}' not found in DB ({tracking_uri}).")
        return

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        print(f"Error: No runs found in experiment '{experiment_name}'.")
        return

    selected_run = None
    if target_run_name:
        for r in runs:
            r_name = r.data.tags.get("mlflow.runName", "")
            if r_name == target_run_name:
                selected_run = r
                break
    
    if selected_run is None:
        selected_run = runs[0]
        print(f"'evaluate_run' not found. Using newest run: {selected_run.info.run_id}")

    run_id = selected_run.info.run_id
    run_name = selected_run.data.tags.get("mlflow.runName", run_id)

    print(f"Load gradient metrics for run '{run_name}' from experiment '{experiment_name}'")
    df_run = load_run_metrics_as_df(client, run_id)

    if df_run.empty:
        print(f"Error: No similarities found in '{run_name}'.")
        return

    if not validate_all_tasks_present(df_run, ALL_TASKS):
        sys.exit(1)

    save_path = os.path.join(output_dir, f"{run_name}_gradient_matrix.png")
    plot_gradient_matrix(df_run, ALL_TASKS, save_path, run_name=run_name)  # type: ignore


if __name__ == "__main__":
    main()