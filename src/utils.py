"""Helper functions for the MTL training and evaluation.
 
Contains the config loading (command line, JSON config and LibMTL YAML config),
the preparation of the LibMTL arguments, the task components (metrics, loss
function and metric weights) and an MLflow logger for the training.
"""


import os
import mlflow
import json
import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime
from LibMTL.config import _display
from metrics import (F1Metrics, F1RegressionMetrics, PPRCMMetrics,
                            F1BinaryMetrics, SegmentationMetrics,
                            DistanceRegressionMetrics, DirectionRegressionMetrics)
from MTL.losses import (get_loss_function)
from LibMTL._record import _PerformanceMeter
from torch.utils.tensorboard import SummaryWriter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args(parser):
    """Add the fixed training arguments to the parser and load the full config.
 
    Only --config is really needed on the command line. All other values are
    defaults that are overwritten by the JSON config and the YAML config (see
    load_parameters).
 
    Returns the final parameters as an argparse.Namespace.
    """
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "MTL/default_config.json"), help="Path to config.json file")
    parser.add_argument('--epochs', default=150, type=int, help='Max number of epochs')
    parser.add_argument('--warmup_epochs', default=15, type=int, help='Number of epochs for the warmup')
    parser.add_argument('--eta_min', type=float, default=1e-7, help='Minimum learning rate for cosine annealing')
    parser.add_argument('--start_factor', type=float, default=0.001, help='Warmup starts with --lr * --start_factor')
    parser.add_argument('--end_factor', type=float, default=1.0, help='Warmup ends with --lr * --end_factor')
    parser.add_argument('--encoder_lr_factor', type=float, default=0.1, help='Encoder LR is multiplied with this factor to reduce LR for the Encoder')
    parser.add_argument('--poly_power', type=float, default=1.0, help='Defines how the LR decreases when warmup_diff_poly scheduler is selected: ' \
    '1 = linear decay, >1 = LR decreases fast at the start and slower at the end, <1 = LR decreases slow at the start and faster at the end')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for the DataLoader')
    parser.add_argument('--pretrained', type=str, default="True", help='Use ImageNet pretrained weights')
    parser.add_argument('--dilate_scale', default=8, type=int, choices=[8, 16], help='Dilate scale for dilated ResNet')
    parser.add_argument('--train_bs', default=12, type=int, help='Batch size for training')
    parser.add_argument('--test_bs', default=12, type=int, help='Batch size for testing and validating')
    params = parser.parse_args()
    return load_parameters(params)


def load_parameters(params):
    """Load the JSON config and the LibMTL YAML config into params.
 
    The "mode" in the JSON config decides what happens:
    - train: a new run folder run_{timestamp} is created in run_dir. The used JSON
      and YAML configs are saved in it, so the run can be evaluated later.
    - eval: the config of the run evaluate_run is loaded, and the checkpoint
      evaluate_checkpoint (default best.pt) of this run is used as load_path.
 
    The YAML values are set first, then the JSON values (nested keys are
    flattened), so the JSON config wins. img_size is converted to an int and bool
    fields given as strings are converted to bools. Raises ValueError if
    evaluate_run is missing in eval mode.
 
    Returns the updated params.
    """
    base_config_path = Path(params.config)
    if not base_config_path.is_absolute():
        base_config_path = PROJECT_ROOT / base_config_path

    with open(base_config_path, "r") as f:
        initial_config_data = json.load(f)

    requested_mode = initial_config_data.get("mode", "train")
    config_data = initial_config_data.copy()

    if requested_mode == "eval":
        eval_run_name = config_data.get("evaluate_run")
        if not eval_run_name:
            raise ValueError(
                "For 'eval' mode 'evaluate_run' is required."
            )

        configured_run_dir = Path(config_data.get("run_dir"))
        if not configured_run_dir.is_absolute():
            configured_run_dir = PROJECT_ROOT / configured_run_dir

        run_path = configured_run_dir / eval_run_name

        eval_config_path = run_path / "config.json"
        if eval_config_path.exists():
            with open(eval_config_path, "r") as f:
                loaded_run_config = json.load(f)
                config_data.update(loaded_run_config)
                config_data["mode"] = requested_mode

        yaml_rel_path = config_data.get("model_configuration", {}).get(
            "detailed_libmtl_configuration", ""
        )
        yaml_path = (
            run_path / yaml_rel_path
            if (run_path / yaml_rel_path).exists()
            else run_path / "detailed_libmtl_configuration.yaml"
        )

        target_save_path = str(run_path)

        ckpt_name = config_data.get("evaluate_checkpoint", "best.pt")
        ckpt_path = run_path / ckpt_name
        if not ckpt_path.exists():
            available = sorted(p.name for p in run_path.glob("*.pt"))
            print(f"[WARN] Checkpoint '{ckpt_name}' not found in {run_path}. "
                  f"Available: {available}")
        target_load_path = str(ckpt_path)

    else:  # requested_mode == 'train'
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir_name = f"run_{timestamp}"

        configured_run_dir = Path(config_data.get("run_dir", "training_runs"))
        if not configured_run_dir.is_absolute():
            configured_run_dir = PROJECT_ROOT / configured_run_dir

        run_path = configured_run_dir / run_dir_name

        yaml_rel_path = config_data.get("model_configuration", {}).get(
            "detailed_libmtl_configuration", ""
        )
        yaml_path = PROJECT_ROOT / yaml_rel_path

        target_save_path = str(run_path)
        target_load_path = None

    yaml_data = {}
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

    for key, value in yaml_data.items():
        setattr(params, key, value)

    json_exclude = {"task_definition"}
    if requested_mode == "train":
        json_exclude.add("evaluate_run")

    flat_json = flatten_dict(config_data, exclude_keys=json_exclude)
    for key, value in flat_json.items():
        if value is not None:
            setattr(params, key, value)

    params.mode = requested_mode

    if "task_definition" in config_data:
        params.task_definition = config_data["task_definition"]

    if getattr(params, "img_size", None) is not None:
        if isinstance(params.img_size, str) and " " in params.img_size:
            params.img_size = params.img_size.split()

        if isinstance(params.img_size, (list, tuple)):
            params.img_size = int(params.img_size[-1])
        else:
            params.img_size = int(params.img_size)
    else:
        params.img_size = None

    bool_fields = ["pretrained", "multi_input", "rep_grad", "ep_grad"]
    for field in bool_fields:
        if hasattr(params, field):
            value = getattr(params, field)
            if isinstance(value, str):
                setattr(
                    params,
                    field,
                    value.lower() in ("true", "1", "t", "y", "yes"),
                )

    params.save_path = target_save_path
    params.load_path = target_load_path

    if requested_mode == "train":
        run_path.mkdir(parents=True, exist_ok=True)

        run_config_data = json.loads(json.dumps(config_data))
        run_config_data["evaluate_run"] = run_path.name
        if "model_configuration" in run_config_data:
            run_config_data["model_configuration"][
                "detailed_libmtl_configuration"
            ] = "detailed_libmtl_configuration.yaml"

        with open(run_path / "config.json", "w") as f:
            json.dump(run_config_data, f, indent=2)

        run_yaml_data = yaml_data.copy()
        for key in run_yaml_data.keys():
            if hasattr(params, key):
                run_yaml_data[key] = getattr(params, key)

        with open(run_path / "detailed_libmtl_configuration.yaml", "w") as f:
            yaml.dump(
                run_yaml_data, f, default_flow_style=False, sort_keys=False
            )

    return params



def flatten_dict(d, exclude_keys=None):
    """Flatten a nested dict into a dict with one level.
 
    Keys in exclude_keys are skipped together with their nested values. Nested keys
    keep only their own name, so equal names in different sub-dicts overwrite each
    other.
 
    Returns the flat dict.
    """
    if exclude_keys is None:
        exclude_keys = set()

    flat = {}
    for k, v in d.items():
        if k in exclude_keys:
            continue
        if isinstance(v, dict):
            flat.update(flatten_dict(v, exclude_keys))
        else:
            flat[k] = v
    return flat



def prepare_args(params):
    """Prepare the arguments for the weighting, architecture, optimizer and scheduler.
 
    Adapted from LibMTL. Checks that all parameters needed by the selected weighting
    method and architecture are set, and prints the configuration. Raises ValueError
    for unknown or incomplete settings.
 
    Returns (kwargs, optim_param, scheduler_param).
    """
    kwargs = {'weight_args': {}, 'arch_args': {}}
    if params.weighting in ['EW', 'UW', 'GradNorm', 'GLS', 'RLW', 'MGDA', 'IMTL',
                            'PCGrad', 'GradVac', 'CAGrad', 'GradDrop', 'DWA',
                            'Nash_MTL', 'MoCo', 'Aligned_MTL', 'DB_MTL', 'STCH',
                            'ExcessMTL', 'FairGrad', 'FAMO', 'MoDo', 'SDMGrad', 'UPGrad']:
        if params.weighting in ['DWA']:
            if params.T is not None:
                kwargs['weight_args']['T'] = params.T
            else:
                raise ValueError('DWA needs keywaord T')
        elif params.weighting in ['GradNorm']:
            if params.alpha is not None:
                kwargs['weight_args']['alpha'] = params.alpha
                kwargs['alpha'] = params.alpha
            else:
                raise ValueError('GradNorm needs keywaord alpha')
        elif params.weighting in ['MGDA']:
            if params.mgda_gn is not None:
                if params.mgda_gn in ['none', 'l2', 'loss', 'loss+']:
                    kwargs['weight_args']['mgda_gn'] = params.mgda_gn
                else:
                    raise ValueError('No support mgda_gn {} for MGDA'.format(params.mgda_gn))
            else:
                raise ValueError('MGDA needs keywaord mgda_gn')
        elif params.weighting in ['GradVac']:
            if params.GradVac_beta is not None:
                kwargs['weight_args']['GradVac_beta'] = params.GradVac_beta
                kwargs['weight_args']['GradVac_group_type'] = params.GradVac_group_type
            else:
                raise ValueError('GradVac needs keywaord beta')
        elif params.weighting in ['GradDrop']:
            if params.leak is not None:
                kwargs['weight_args']['leak'] = params.leak
            else:
                raise ValueError('GradDrop needs keywaord leak')
        elif params.weighting in ['CAGrad']:
            if params.calpha is not None and params.rescale is not None:
                kwargs['weight_args']['calpha'] = params.calpha
                kwargs['weight_args']['rescale'] = params.rescale
            else:
                raise ValueError('CAGrad needs keywaord calpha and rescale')
        elif params.weighting in ['Nash_MTL']:
            if params.update_weights_every is not None and params.optim_niter is not None and params.max_norm is not None:
                kwargs['weight_args']['update_weights_every'] = params.update_weights_every
                kwargs['weight_args']['optim_niter'] = params.optim_niter
                kwargs['weight_args']['max_norm'] = params.max_norm
            else:
                raise ValueError('Nash_MTL needs update_weights_every, optim_niter, and max_norm')
        elif params.weighting in ['MoCo']:
            kwargs['weight_args']['MoCo_beta'] = params.MoCo_beta
            kwargs['weight_args']['MoCo_beta_sigma'] = params.MoCo_beta_sigma
            kwargs['weight_args']['MoCo_gamma'] = params.MoCo_gamma
            kwargs['weight_args']['MoCo_gamma_sigma'] = params.MoCo_gamma_sigma
            kwargs['weight_args']['MoCo_rho'] = params.MoCo_rho
        elif params.weighting in ['DB_MTL']:
            kwargs['weight_args']['DB_beta'] = params.DB_beta
            kwargs['weight_args']['DB_beta_sigma'] = params.DB_beta_sigma
        elif params.weighting in ['STCH']:
            kwargs['weight_args']['STCH_mu'] = params.STCH_mu
            kwargs['weight_args']['STCH_warmup_epoch'] = params.STCH_warmup_epoch
        elif params.weighting in ['ExcessMTL']:
            kwargs['weight_args']['robust_step_size'] = params.robust_step_size
        elif params.weighting in ['FairGrad']:
            kwargs['weight_args']['FairGrad_alpha'] = params.FairGrad_alpha
        elif params.weighting in ['FAMO']:
            kwargs['weight_args']['FAMO_w_lr'] = params.FAMO_w_lr
            kwargs['weight_args']['FAMO_w_gamma'] = params.FAMO_w_gamma
        elif params.weighting in ['MoDo']:
            kwargs['weight_args']['MoDo_gamma'] = params.MoDo_gamma
            kwargs['weight_args']['MoDo_rho'] = params.MoDo_rho
        elif params.weighting in ['SDMGrad']:
            kwargs['weight_args']['SDMGrad_lamda'] = params.SDMGrad_lamda
            kwargs['weight_args']['SDMGrad_niter'] = params.SDMGrad_niter
        elif params.weighting in ['UPGrad']:
            kwargs['weight_args']['UPGrad_norm_eps'] = params.UPGrad_norm_eps
            kwargs['weight_args']['UPGrad_reg_eps'] = params.UPGrad_reg_eps
    elif params.weighting in ['MOML', 'FORUM', 'AutoLambda']:
        kwargs['weight_args']['outer_lr'] = params.outer_lr
        kwargs['weight_args']['inner_step'] = params.inner_step
        if params.weighting in ['FORUM']:
            kwargs['weight_args']['FORUM_phi'] = params.FORUM_phi
            kwargs['weight_args']['inner_lr'] = params.inner_lr
        elif params.weighting in ['MOML']:
            kwargs['weight_args']['inner_lr'] = params.inner_lr
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting))

    if params.arch in ['HPS', 'Cross_stitch', 'MTAN', 'CGC', 'PLE', 'MMoE', 'DSelect_k', 'DIY', 'LTB']:
        if params.arch in ['CGC', 'PLE', 'MMoE', 'DSelect_k']:
            kwargs['arch_args']['img_size'] = tuple(params.img_size)
            kwargs['arch_args']['num_experts'] = [int(num) for num in params.num_experts]
        if params.arch in ['DSelect_k']:
            kwargs['arch_args']['kgamma'] = params.kgamma
            kwargs['arch_args']['num_nonzeros'] = params.num_nonzeros
    else:
        raise ValueError('No support architecture method {}'.format(params.arch))

    if params.optim in ['adam', 'sgd', 'adagrad', 'rmsprop', 'adamw']:
        optim_param = {'optim': params.optim, 'lr': params.lr, 'weight_decay': params.weight_decay}
        if params.optim == 'sgd':
            optim_param['momentum'] = params.momentum
    else:
        raise ValueError('No support optim method {}'.format(params.optim))

    if params.scheduler is not None:
        if params.scheduler in ['step', 'cos', 'exp', 'warmup_cos', 'diff_cos', 'warmup_diff_poly']:
            if params.scheduler == 'step':
                scheduler_param = {'scheduler': 'step', 'step_size': params.step_size, 'gamma': params.gamma}
            else:
                scheduler_param = {'scheduler': params.scheduler}
        else:
            raise ValueError('No support scheduler method {}'.format(params.scheduler))
    else:
        scheduler_param = None

    _display(params, kwargs, optim_param, scheduler_param)

    return kwargs, optim_param, scheduler_param




def get_task_components(task_name: str, params):
    """Create the metrics, the loss function and the metric weights for one task.
 
    The loss function is read from task_definition in the config ("default" if it
    is not set). The weight list says for each metric if higher is better (1) or
    lower is better (0), as LibMTL expects it. Raises ValueError for an unknown task.
 
    Returns a dict with the keys metrics, metrics_fn, loss_fn and weight.
    """
    task_config = getattr(params, 'task_definition', {}).get(task_name, {})

    class TaskParams:
        def __init__(self, loss_name):
            self.loss_function = loss_name

    loss_name = task_config.get('loss_function', 'default')
    task_p = TaskParams(loss_name)

    if task_name == "bbch_classification":
        return {
            'metrics': ["Macro_F1"],
            'metrics_fn': F1Metrics(num_classes=8),
            'loss_fn': get_loss_function("bbch_classification", task_p),
            'weight': [1]
        }
    elif task_name == "bbch_regression":
        return {
            'metrics': ["Macro_F1"],
            'metrics_fn': F1RegressionMetrics(bbch_min=params.bbch_min, bbch_max=params.bbch_max),
            'loss_fn': get_loss_function("bbch_regression", task_p),
            'weight': [1]
        }
    elif task_name == "pprcm_regression":
        return {
            'metrics': ["RMSE", "MAE"],
            'metrics_fn': PPRCMMetrics(ppr_min=params.ppr_min, ppr_max=params.ppr_max),
            'loss_fn': get_loss_function("pprcm_regression", task_p),
            'weight': [0, 0]
        }
    elif task_name == "damaged_classification":
        return {
            'metrics': ["Macro_F1"],
            'metrics_fn': F1BinaryMetrics(),
            'loss_fn': get_loss_function("damaged_classification", task_p),
            'weight': [1]
        }
    elif task_name == "semantic_segmentation":
        return {
            'metrics': ["mIoU", "Macro_F1", "main_plant_mIoU"],
            'metrics_fn': SegmentationMetrics(num_classes=4, ignore_index=255),
            'loss_fn': get_loss_function("semantic_segmentation", task_p),
            'weight': [1, 1, 1]
        }
    elif task_name == "distance_regression":
        return {
            'metrics': ["rmse", "mae", "p_mae", "p_rmse"],
            'metrics_fn': DistanceRegressionMetrics(min_valid=-1.5),
            'loss_fn': get_loss_function("distance_regression", task_p),
            'weight': [0, 0, 0, 0]
        }
    elif task_name == "direction_regression":
        return {
            'metrics': ["vec_error", "ang_error", "magni_error"],
            'metrics_fn': DirectionRegressionMetrics(min_valid=-1.5),
            'loss_fn': get_loss_function("direction_regression", task_p),
            'weight': [0, 0, 0]
        }
    else:
        raise ValueError(f"Unknown task: {task_name}")



class MLflowPerformanceMeter(_PerformanceMeter):
    """LibMTL performance meter that also logs the training to MLflow.
 
    Per epoch it logs the losses, the metrics, the epoch time, the learning rates and
    the gradient similarities between the tasks. The main task (bbch_regression) is
    logged in its own section, all other tasks as auxiliary tasks. In train mode with
    similarities, one row per epoch is also added to task_similarities.csv in log_dir.
    """

    def __init__(self, task_dict, multi_input, log_dir=None, base_result=None):
        super().__init__(task_dict, multi_input, base_result)
        self.log_dir = log_dir
        if self.log_dir:
            self.csv_path = os.path.join(self.log_dir, 'task_similarities.csv')
        else:
            self.csv_path = None

        self.main_task = "bbch_regression"

    def display(self, mode, epoch, model=None, optimizer=None, similarities=None):
        """Print the results of one epoch (like LibMTL) and log them to MLflow.
 
        Nothing is logged if epoch is None.
        """
        super().display(mode, epoch)

        if epoch is None:
            return

        metrics_to_log = {}
        csv_row = {'epoch': epoch, 'mode': mode}

        for tn, task in enumerate(self.task_name):
            loss_val = float(self.loss_item[tn])

            if task == self.main_task:
                metrics_to_log[f"Main_Task_Loss/loss_{mode}"] = loss_val
            else:
                metrics_to_log[f"Aux_Tasks_Losses/{task}_{mode}"] = loss_val

            csv_row[f'loss_{task}'] = loss_val

        for task in self.task_name:
            is_main = (task == self.main_task)

            for i, metric_value in enumerate(self.results[task]):
                metric_name = self.task_dict[task]['metrics'][i]
                val = float(metric_value)

                if is_main:
                    metrics_to_log[f"Main_Task_Accuracy/{metric_name}_{mode}"] = val
                else:
                    metrics_to_log[f"Aux_Tasks_Accuracies/{task}__{metric_name}_{mode}"] = val

                csv_row[f'metric_{task}_{metric_name}'] = val

        if hasattr(self, "end_time") and hasattr(self, "beg_time"):
            metrics_to_log[f"Epoch_Time/{mode}"] = float(self.end_time - self.beg_time)

        if optimizer is not None and mode == 'train':
            for i, param_group in enumerate(optimizer.param_groups):
                group_name = param_group.get('name', f"group_{i}")
                lr_val = float(param_group['lr'])
                metrics_to_log[f"Learning_Rates/{group_name}"] = lr_val
                csv_row[f'lr_{group_name}'] = lr_val

        if similarities is not None and mode == 'train':
            task_keys = list(self.task_name)
            num_tasks = len(task_keys)

            for i in range(num_tasks):
                for j in range(i + 1, num_tasks):
                    t1, t2 = task_keys[i], task_keys[j]

                    cos_val = float(similarities['cosine'][i, j].item())
                    mag_val = float(similarities['magnitude'][i, j].item())

                    metrics_to_log[f"Similarities_Cosine/{t1}_vs_{t2}"] = cos_val
                    metrics_to_log[f"Similarities_Magnitude/{t1}_vs_{t2}"] = mag_val

                    csv_row[f'cos_{t1}_vs_{t2}'] = cos_val
                    csv_row[f'mag_{t1}_vs_{t2}'] = mag_val

            if self.csv_path:
                df_new = pd.DataFrame([csv_row])
                if not os.path.isfile(self.csv_path):
                    df_new.to_csv(self.csv_path, index=False)
                else:
                    df_new.to_csv(self.csv_path, mode='a', header=False, index=False)

        mlflow.log_metrics(metrics_to_log, step=epoch)