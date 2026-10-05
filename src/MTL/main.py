"""Train or evaluate the MTL model.
 
The "mode" in the config decides what happens (see load_parameters in utils.py):
- train: train a new model in a new run folder and log everything to MLflow.
- eval: load a checkpoint of an existing run and evaluate it on the test set.
 
The MLflow database (mlflow.db) is in the parent folder of the run folders and
is used with the experiment_name from the config.
 
Usage:
    python3 src/MTL/main.py --config <config.json>
 
    The MLflow results can be viewed with:
    mlflow ui --backend-store-uri sqlite:///<training_runs_dir>/mlflow.db
"""


import sys
import warnings
from pathlib import Path
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # which GPU should be used

from LibMTL.utils import set_random_seed
from LibMTL.config import LibMTL_args

# make src importable, load the config and set the seed
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import parse_args, prepare_args, get_task_components
PARAMS = parse_args(LibMTL_args)
set_random_seed(PARAMS.seed)

from encoders import get_encoder
from decoders import get_decoder
from bbch_dataset import get_dataloaders
from bbch_trainer import BBCH_Trainer
import torch.nn as nn
import torch
import mlflow


def main():
    """Build the model for the selected tasks and train or evaluate it.
 
    One decoder and one set of task components (loss and metrics) is created for
    each task in selected_tasks. In eval mode, a missing checkpoint only gives a
    warning, so the model is then tested with untrained decoders.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
    warnings.filterwarnings("ignore", message=".*ShiftScaleRotate is a special case of Affine.*")

    run_dir_path = Path(PARAMS.save_path)
    run_name = run_dir_path.name  # e.g. "run_2026-08-25_12-30-00"
    experiment_name = PARAMS.experiment_name

    training_runs_dir = run_dir_path.parent
    db_path = training_runs_dir / "mlflow.db"
    
    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("run_dir", str(run_dir_path))
        mlflow.set_tag("mode", PARAMS.mode)

        kwargs, optim_param, scheduler_param = prepare_args(PARAMS)

        train_loader, val_loader, test_loader = get_dataloaders(PARAMS)

        def encoder_class():
            return get_encoder(PARAMS)

        decoders = nn.ModuleDict()
        task_dict = {}

        selected_tasks = getattr(PARAMS, 'selected_tasks')
        for task in selected_tasks:
            decoder_type = PARAMS.task_definition.get(task, {}).get('decoder')
            
            num_classes = None
            if "classification" in task:
                num_classes = 8 if "bbch" in task else 2
            elif task == "semantic_segmentation":
                num_classes = 4

            decoders[task] = get_decoder(task_name=task, num_classes=num_classes, decoder_name=decoder_type)
            
            task_dict[task] = get_task_components(task, PARAMS)

        model = BBCH_Trainer(
                task_dict=task_dict,
                weighting=PARAMS.weighting,
                architecture=PARAMS.arch,
                encoder_class=encoder_class,
                decoders=decoders,
                rep_grad=PARAMS.rep_grad,
                multi_input=PARAMS.multi_input,
                optim_param=optim_param,
                scheduler_param=scheduler_param,
                save_path=PARAMS.save_path,
                load_path=PARAMS.load_path,
                params=PARAMS,
                **kwargs
            )
        
        if PARAMS.mode == "train":
            model.train(
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                test_dataloaders=test_loader,
                epochs=PARAMS.epochs
            )
        elif PARAMS.mode == "eval":
            checkpoint_path = Path(PARAMS.load_path)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=model.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"WARNING: missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"WARNING: unexpected keys: {unexpected_keys}")
                
            else:
                warnings.warn(f"No checkpoint found under {checkpoint_path}!")

            model.model.eval()
            
            with torch.no_grad():
                model.test(test_loader, mode='test')

  
if __name__ == "__main__":
    main()