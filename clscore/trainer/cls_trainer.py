import os
import sys
from typing import Any, Optional
from torchmetrics import MeanMetric, Accuracy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from apps.trainer.base import EpochTrainer
from apps.utils.dist import is_master, sync_tensor
import math
from apps.utils.model import list_join, list_mean

__all__ = ["ClsTrainer"]


class ClsTrainer(EpochTrainer):
    def __init__(
        self,
        path: str,
        model: nn.Module,
        data_provider,
        auto_restart_thresh: Optional[float] = None,
        logger =None
    ) -> None:
        super().__init__(
            path=path,
            model=model,
            data_provider=data_provider,
            logger=logger
        )
        self.auto_restart_thresh = auto_restart_thresh
        self.test_criterion = nn.CrossEntropyLoss()
    def initialize_metrics(self):
        num_classes = self.data_provider.main_classes
        # Initialize main metrics for classification
        metrics = {
            "main": {
                "loss": MeanMetric().cuda(),  # Average loss
                "top1": Accuracy(top_k=1, num_classes=num_classes).cuda(),  # Top-1 Accuracy
            }
        }
        if self.data_provider.main_classes >= 100:
            metrics["main"]["top5"] = Accuracy(top_k=5, num_classes=num_classes).cuda()
        if self.run_config.mesa is not None:
            metrics["mesa"] = {
                "loss": MeanMetric().cuda(),
            }

        return metrics
    
    def reset_metrics(self):
        for key, metric_dict in self.metrics.items():
            for metric_name, metric in metric_dict.items():
                metric.reset()

    def compute_metrics(self,metrics, prefix):
        final_metrics = {}
        for key, value in metrics.items():
            new_key = f'{prefix}_{key}'
            if isinstance(value, dict):
                final_metrics.update(self.compute_metrics(value, prefix=new_key))
            else:
                final_metrics[new_key] = value.compute()
        return final_metrics
    
    def _validate(self, model, data_loader, epoch) -> dict[str, Any]:
        self.reset_metrics()
        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc=f"Validate Epoch #{epoch + 1}",
                disable=not is_master(),
                file=sys.stdout,
            ) as t:
                for samples in data_loader:
                    images = samples.get("image", None)
                    labels = samples.get("labels", None)
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = model(images)
                    if isinstance(output, dict):
                        main_output = output["main"]
                    else:
                        main_output = output


                    main_loss = self.test_criterion(main_output, labels)
                    self.metrics["main"]["loss"].update(main_loss.item(), images.shape[0])

                    self.metrics["main"]["top1"].update(main_output, labels)
                    if self.data_provider.main_classes >= 100:
                        self.metrics["main"]["top5"].update(main_output, labels)


                    t.set_postfix(
                        {
                           "loss": self.metrics["main"]["loss"].compute().item() if "loss" in self.metrics["main"] else None,
                            "top1": self.metrics["main"]["top1"].compute().item() if "top1" in self.metrics["main"] else None,
                             "bs": images.shape[0],
                        }
                    )
                    t.update()
        final_metrics = self.compute_metrics(self.metrics,"val")
        return final_metrics
    


    def before_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        images = feed_dict["data"].cuda()
        labels = feed_dict["label"].cuda()

        # label smooth
        #labels = label_smooth(labels, self.data_provider.n_classes, self.run_config.label_smooth)

        # mixup
        '''
        if self.run_config.mixup_config is not None:
            # choose active mixup config
            mix_weight_list = [mix_list[2] for mix_list in self.run_config.mixup_config["op"]]
            active_id = torch_random_choices(
                list(range(len(self.run_config.mixup_config["op"]))),
                weight_list=mix_weight_list,
            )
            active_id = int(sync_tensor(active_id, reduce="root"))
            active_mixup_config = self.run_config.mixup_config["op"][active_id]
            mixup_type, mixup_alpha = active_mixup_config[:2]

            lam = float(torch.distributions.beta.Beta(mixup_alpha, mixup_alpha).sample())
            lam = float(np.clip(lam, 0, 1))
            lam = float(sync_tensor(lam, reduce="root"))

            images, labels = apply_mixup(images, labels, lam, mixup_type)
        '''
        return {
            "data": images,
            "label": labels,
        }

    def _run_step(self, feed_dict: dict[str, Any]) -> dict[str, Any]:
        images = feed_dict["data"]
        labels = feed_dict["label"]

        # setup mesa
        if self.run_config.mesa is not None and self.run_config.mesa["thresh"] <= self.run_config.progress:
            ema_model = self.ema.shadows
            with torch.inference_mode():
                ema_output = ema_model(images).detach()
                if isinstance(ema_output, dict):
                    ema_main_output = ema_output["main"]
                else:
                    ema_main_output = ema_output
            ema_main_output = torch.clone(ema_main_output)
        else:
            ema_main_output = None

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            output = self.model(images)
            if isinstance(output, dict):
                main_output = output["main"]
            else:
                main_output = output
            loss = self.train_criterion(main_output, labels)
            # mesa loss
            if ema_output is not None:
                mesa_loss = self.train_criterion(output, ema_main_output)
                loss = loss + self.run_config.mesa["ratio"] * mesa_loss
        if not math.isfinite(loss.item()): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss.item()))
            assert math.isfinite(loss.item())
        self.scaler.scale(loss).backward()

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        self.reset_metrics()

        with tqdm(
            total=len(self.data_provider.train),
            desc="Train Epoch #{}".format(epoch + 1),
            disable=not is_master(),
            file=sys.stdout,
        ) as t:
            for samples in self.data_provider.train:
                # preprocessing
                feed_dict = self.before_step(samples)
                # clear gradient
                self.optimizer.zero_grad()
                # forward & backward
                self._run_step(feed_dict)
                # update: optimizer, lr_scheduler
                self.after_step()

                # tqdm
                postfix_dict = {
                    "loss": self.metrics["main"]["loss"].compute().item() if "loss" in self.metrics["main"] else None,
                     "top1": self.metrics["main"]["top1"].compute().item() if "top1" in self.metrics["main"] else None,
                    "bs": samples["image"].shape[0],
                    "lr": list_join(
                        sorted(set([group["lr"] for group in self.optimizer.param_groups])),
                        "#",
                        "%.1E",
                    ),
                    "progress": self.run_config.progress,
                }
                t.set_postfix(postfix_dict)
                t.update()

    def train(self, trials=0, save_freq=1) -> None:

        for epoch in range(self.start_epoch, self.run_config.n_epochs + self.run_config.warmup_epochs):
            train_info_dict = self.train_one_epoch(epoch)
            # eval
            val_info_dict = self.validate(epoch=epoch)
            if self.logger:
                self.logger.log_metrics(metrics=val_info_dict, step=epoch + self.run_config.eval_interval)
            avg_top1 = val_info_dict["val_main_top1"].item()
            is_best = avg_top1 > self.best_val
            self.best_val = max(avg_top1, self.best_val)

            if self.auto_restart_thresh is not None:
                if self.best_val - avg_top1 > self.auto_restart_thresh:
                    self.write_log(f"Abnormal accuracy drop: {self.best_val} -> {avg_top1}")
                    self.load_model(os.path.join(self.checkpoint_path, "model_best.pt"))
                    return self.train(trials + 1, save_freq)

            # save model
            if (epoch + 1) % save_freq == 0 or (is_best and self.run_config.progress > 0.8):
                self.save_model(
                    only_state_dict=False,
                    epoch=epoch,
                    model_name="model_best.pt" if is_best else "checkpoint.pt",
                )