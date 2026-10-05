"""LibMTL trainer for the BBCH multi-task learning."""


import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
from LibMTL import Trainer
from utils import MLflowPerformanceMeter  # type: ignore


class BBCH_Trainer(Trainer):
    """LibMTL trainer for any combination of the selected tasks.
 
    Changes to the LibMTL Trainer:
    - The encoder gets a lower learning rate than the decoders (encoder_lr_factor).
    - Own learning rate schedulers (warmup_cos, diff_cos, warmup_diff_poly).
    - Predictions of the pixel-wise tasks are resized to the size of the targets,
      and the direction vectors are normalized to length 1.
    - In each epoch, the cosine and magnitude similarity of the encoder gradients
      of all task pairs are computed and logged.
    - Checkpoints latest.pt, best.pt and last.pt, and early stopping.
    """

    def __init__(self, task_dict, weighting, architecture, encoder_class,
                 decoders, rep_grad, multi_input, optim_param,
                 scheduler_param, **kwargs):

        self.params = kwargs.get('params')
        self.optim_param = optim_param
        self.scheduler_param = scheduler_param

        super(BBCH_Trainer, self).__init__(
            task_dict=task_dict,
            weighting=weighting,
            architecture=architecture,
            encoder_class=encoder_class,
            decoders=decoders,
            rep_grad=rep_grad,
            multi_input=multi_input,
            optim_param=optim_param,
            scheduler_param=scheduler_param,
            **kwargs
        )

        self.meter = MLflowPerformanceMeter(
            self.task_dict,
            self.multi_input
        )

        self.spatial_tasks = ["semantic_segmentation", "distance_regression", "direction_regression"]
        self._last_improvement = float('-inf')
        self._last_scores = {}


    def _prepare_optimizer(self, optim_param, scheduler_param):
        """Create the optimizer and the learning rate scheduler.
 
        The encoder gets the learning rate lr * encoder_lr_factor and the decoders
        get lr. 'adamw' and 'adam' are used as given, all other optimizer names use
        SGD. Schedulers:
        - warmup_cos: linear warmup for warmup_epochs, then cosine decay to eta_min.
        - diff_cos: cosine decay over all epochs to eta_min.
        - warmup_diff_poly: linear warmup, then polynomial decay with poly_power.
        """
        optim_type = optim_param['optim']
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        base_lr = optim_param['lr']

        backbone_params = self.model.encoder.parameters()
        decoder_params = self.model.decoders.parameters()

        params_groups = [
            {
                'params': backbone_params,
                'lr': base_lr * getattr(self.params, 'encoder_lr_factor', 0.1),
                'name': 'backbone'
            },
            {
                'params': decoder_params,
                'lr': base_lr,
                'name': 'decoders'
            }
        ]

        if optim_type == 'adamw':
            self.optimizer = torch.optim.AdamW(params_groups, **optim_arg)
        elif optim_type == 'adam':
            self.optimizer = torch.optim.Adam(params_groups, **optim_arg)
        else:
            self.optimizer = torch.optim.SGD(params_groups, **optim_arg)

        if scheduler_param is not None:
            sched_type = scheduler_param['scheduler']
            if sched_type == 'warmup_cos':
                warmup_epochs = getattr(self.params, 'warmup_epochs', 10)
                total_epochs = self.params.epochs

                s1 = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=getattr(self.params, 'start_factor', 0.001),
                    end_factor=getattr(self.params, 'end_factor', 1.0),
                    total_iters=warmup_epochs
                )
                s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=(total_epochs - warmup_epochs),
                    eta_min=getattr(self.params, 'eta_min', 1e-7)
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[s1, s2],
                    milestones=[warmup_epochs]
                )

            elif sched_type == "diff_cos":
                total_epochs = self.params.epochs
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_epochs,
                    eta_min=getattr(self.params, 'eta_min', 1e-7)
                )

            elif sched_type == "warmup_diff_poly":
                warmup_epochs = getattr(self.params, 'warmup_epochs', 15)
                total_epochs = self.params.epochs

                s1 = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=getattr(self.params, 'start_factor', 0.001),
                    total_iters=warmup_epochs
                )
                s2 = torch.optim.lr_scheduler.PolynomialLR(
                    self.optimizer,
                    total_iters=(total_epochs - warmup_epochs),
                    power=getattr(self.params, 'poly_power', 1.0)
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[s1, s2],
                    milestones=[warmup_epochs]
                )
        else:
            self.scheduler = None


    def process_preds(self, preds, task_name=None, gts=None):
        """Resize the predictions of the pixel-wise tasks to the target size.
 
        The predictions are resized with bilinear interpolation to the size of the GT
        masks (or img_size x img_size if no GT is given). The direction predictions
        are also normalized to length 1. Plant-level predictions are not changed.
 
        Returns the processed predictions: one tensor in multi_input mode, otherwise
        a dict with one tensor per task.
        """
        if self.multi_input:
            if task_name in self.spatial_tasks:
                target_size = gts.shape[-2:] if gts is not None else (self.params.img_size, self.params.img_size)
                if preds.dim() >= 3 and preds.shape[-2:] != target_size:
                    preds = F.interpolate(preds, size=target_size, mode='bilinear', align_corners=False)
                if task_name == "direction_regression":
                    preds = F.normalize(preds, p=2, dim=1)
            return preds
        else:
            processed_preds = {}
            for tn in self.task_name:
                pred = preds[tn]
                if tn in self.spatial_tasks:
                    target_size = gts[tn].shape[-2:] if (gts is not None and tn in gts) else (self.params.img_size, self.params.img_size)
                    if pred.dim() >= 3 and pred.shape[-2:] != target_size:
                        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
                    if tn == "direction_regression":
                        pred = F.normalize(pred, p=2, dim=1)
                processed_preds[tn] = pred
            return processed_preds


    def forward4loss(self, model, inputs, gts, return_preds=False):
        """Run the model and compute the loss of each task.
 
        Returns the losses (one per task), and also the predictions if return_preds
        is True.
        """
        if not self.multi_input:
            preds = model(inputs)
            preds = self.process_preds(preds, gts=gts)
            losses = self._compute_loss(preds, gts)
        else:
            losses = torch.zeros(self.task_num).to(self.device)
            preds = {}
            for tn, task in enumerate(self.task_name):
                inputs_t, gts_t = inputs[task], gts[task]
                preds_t = model(inputs_t, task)
                preds_t = preds_t[task]
                preds_t = self.process_preds(preds_t, task, gts=gts_t)
                losses[tn] = self._compute_loss(preds_t, gts_t, task)
                if return_preds:
                    preds[task] = preds_t

        if return_preds:
            return losses, preds
        else:
            return losses


    def _update_dataset_epoch(self, epoch, train_dataloaders):
        """Pass the epoch to the train datasets (see BBCH_Dataset.set_epoch)."""
        if not self.multi_input:
            loader = train_dataloaders[0] if isinstance(train_dataloaders, list) else train_dataloaders
            if hasattr(loader.dataset, 'set_epoch'):
                loader.dataset.set_epoch(epoch)
        else:
            for task in self.task_name:
                loader = train_dataloaders[task]
                if hasattr(loader.dataset, 'set_epoch'):
                    loader.dataset.set_epoch(epoch)


    def _weights_only_ckpt(self, epoch, val_scores, test_scores, rule):
        """Build a small checkpoint with only the model weights and metadata.
 
        Returns the checkpoint dict.
        """
        return {
            'epoch': int(epoch),
            'selection_rule': rule,
            'model_state_dict': {k: v.detach().cpu().clone()
                                 for k, v in self.model.state_dict().items()},
            'val_scores': val_scores,
            'test_scores': test_scores,
        }


    def _save_weights(self, path, epoch, val_scores, test_scores, rule):
        """Save a weights-only checkpoint (see _weights_only_ckpt) to path."""
        torch.save(self._weights_only_ckpt(epoch, val_scores, test_scores, rule), path)


    def train_singlelevel(self, train_dataloaders, test_dataloaders, epochs, val_dataloaders=None, return_weight=False):
        """Train the model, evaluate it after each epoch and save the checkpoints.
 
        In each batch, the gradient of each task loss w.r.t. the encoder is computed
        with an extra backward pass and summed up over the epoch. From the mean
        gradients, the cosine and magnitude similarity of all task pairs are
        computed (magnitude: 2 |gi| |gj| / (|gi|^2 + |gj|^2), 1 means equal size).
        The model itself is updated with the combined gradient of the weighting
        method.
 
        After each epoch, the model is evaluated on val and test, and the scheduler
        makes a step. Checkpoints:
        - latest.pt: full checkpoint in every epoch, to resume the training.
        - best.pt: weights of the epoch with the best val improvement so far.
        - last.pt: weights of the last epoch.
        The training stops early if the val improvement does not get better for
        early_stopping epochs.
        """
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs

        patience = getattr(self.params, 'early_stopping', 150)
        epochs_without_improvement = 0
        best_val_improvement = -float('inf')
        self.actual_best_epoch = 0
        
        last_epoch_done = -1

        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')

            epoch_loss_sums = np.zeros(self.task_num)
            epoch_task_grads = {task: {} for task in self.task_name}

            self._update_dataset_epoch(epoch, train_dataloaders)

            for batch_index in range(train_batch):
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                else:
                    train_inputs, train_gts = {}, {}
                    for tn, task in enumerate(self.task_name):
                        train_inputs[task], train_gts[task] = self._process_data(train_loader[task])

                train_losses_, train_preds = self.forward4loss(self.model, train_inputs, train_gts, return_preds=True)

                for tn, task in enumerate(self.task_name):
                    self.optimizer.zero_grad(set_to_none=False)
                    train_losses_[tn].backward(retain_graph=True)
                    epoch_loss_sums[tn] += train_losses_[tn].item()

                    for name, param in self.model.encoder.named_parameters():
                        if param.grad is not None:
                            grad_copy = param.grad.detach().cpu().clone()
                            if name not in epoch_task_grads[task]:
                                epoch_task_grads[task][name] = grad_copy
                            else:
                                epoch_task_grads[task][name] += grad_copy

                if not self.multi_input:
                    self.meter.update(train_preds, train_gts)
                else:
                    for tn, task in enumerate(self.task_name):
                        self.meter.update(train_preds[task], train_gts[task], task)

                self.optimizer.zero_grad(set_to_none=False)

                alpha_val = getattr(self.params, 'alpha', 1.5)
                self.model.backward(train_losses_, alpha=alpha_val)
                self.optimizer.step()

            for tn in range(self.task_num):
                self.model.train_loss_buffer[tn, epoch] = epoch_loss_sums[tn] / train_batch

            for task in self.task_name:
                for name in epoch_task_grads[task]:
                    epoch_task_grads[task][name] /= train_batch

            task_keys = list(self.task_name)
            num_tasks = len(task_keys)
            similarities = None
            if num_tasks > 1:
                similarities = {
                    'cosine': torch.zeros((num_tasks, num_tasks)),
                    'magnitude': torch.zeros((num_tasks, num_tasks))
                }
                task_vectors = {}
                for task in task_keys:
                    grads = [epoch_task_grads[task][name].flatten() for name in epoch_task_grads[task]]
                    task_vectors[task] = torch.cat(grads)

                for i in range(num_tasks):
                    for j in range(i, num_tasks):
                        gi = task_vectors[task_keys[i]]
                        gj = task_vectors[task_keys[j]]

                        cos_sim = F.cosine_similarity(gi.unsqueeze(0), gj.unsqueeze(0)).item()
                        norm_i = torch.norm(gi)
                        norm_j = torch.norm(gj)
                        mag_sim = ((2 * norm_i * norm_j) / (norm_i ** 2 + norm_j ** 2 + 1e-8)).item()

                        similarities['cosine'][i, j] = similarities['cosine'][j, i] = cos_sim
                        similarities['magnitude'][i, j] = similarities['magnitude'][j, i] = mag_sim

            self.meter.record_time('end')
            self.meter.get_score()
            self.meter.display(epoch=epoch, mode='train', model=self.model, optimizer=self.optimizer, similarities=similarities)
            self.meter.reinit()

            improvement_detected = False
            val_improvement = float('-inf')
            val_scores = {}
            if val_dataloaders is not None:
                self.meter.has_val = True
                self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
                val_improvement = self._last_improvement
                val_scores = copy.deepcopy(self._last_scores)

                if epoch == 0 or val_improvement > best_val_improvement:
                    best_val_improvement = val_improvement
                    improvement_detected = True

            self.test(test_dataloaders, epoch, mode='test')
            test_scores = copy.deepcopy(self._last_scores)

            if self.scheduler is not None:
                self.scheduler.step()

            checkpoint = {
                'epoch': int(epoch),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'task_gradients': epoch_task_grads,
                'best_val_improvement': float(best_val_improvement)
            }
            torch.save(checkpoint, os.path.join(self.save_path, 'latest.pt'))

            last_epoch_done = epoch

            if improvement_detected:
                self._save_weights(os.path.join(self.save_path, 'best.pt'),
                                   epoch, val_scores, test_scores, rule="best_val")
                print(f'Save Model {epoch} to {self.save_path}/best.pt')
                epochs_without_improvement = 0
                self.actual_best_epoch = epoch
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("\n" + "=" * 40)
                print(f"EARLY STOPPING triggered after {epoch+1} epochs with patience of {patience}")
                print(f"Best validation epoch was {self.actual_best_epoch}")
                break

        self._save_weights(os.path.join(self.save_path, 'last.pt'), last_epoch_done, val_scores, test_scores, rule="last")
        self.meter.display_best_result()


    def train(self, train_dataloaders, test_dataloaders, epochs, val_dataloaders=None, return_weight=False, **kwargs):
        """Start the training with the LibMTL Trainer.train (no own changes)."""
        super().train(train_dataloaders, test_dataloaders, epochs,
                      val_dataloaders=val_dataloaders,
                      return_weight=return_weight, **kwargs)


    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        """Evaluate the model on the val or test set.
 
        The losses and metrics are recorded and displayed by the meter. The scores of
        all tasks are stored in self._last_scores and the improvement in
        self._last_improvement.
 
        Returns the improvement if return_improvement is True, otherwise None.
        """
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds, gts=test_gts)
                    _ = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred, task, gts=test_gt)
                        _ = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)

        self.meter.record_time('end')
        self.meter.get_score()

        self._last_scores = {t: [float(v) for v in self.meter.results[t]]
                             for t in self.task_name}

        self.meter.display(epoch=epoch, mode=mode, model=self.model)
        improvement = self.meter.improvement
        self._last_improvement = float(improvement) if improvement is not None else float('-inf')
        self.meter.reinit()
        if return_improvement:
            return improvement