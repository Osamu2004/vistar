import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from apps.trainer.base import StepTrainer
from apps.utils.metric import AverageMeter
from apps.utils.dist import is_master, sync_tensor
from apps.builder import make_loss
from apps.utils.model import list_join, list_mean
from torchmetrics import MeanMetric, JaccardIndex, Precision, Recall, F1Score
from apps.builder import make_loss
__all__ = ["CDTrainer"]
import pdb

class SegTrainer(StepTrainer):
    def __init__(
        self,
        path: str,
        model: nn.Module,
        data_provider,
        auto_restart_thresh: float or None = None,
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
    def initialize_metrics(self,):
        """
        使用 torchmetrics 初始化评价指标
        :param num_classes: 数据集类别数量，用于 IoU 等多分类指标计算
        :param data_loader: PyTorch 数据加载器
        :return: 包含指标的字典
        """
        num_classes = self.data_provider.main_classes
        # 初始化主要评价指标
        metrics = {
            "main": {
                "loss": MeanMetric().cuda(),  # 平均损失
                "iou": JaccardIndex(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes).cuda(),
                "precision": Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
                "recall": Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
                "f1": F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
            }
        }
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
        """
        遍历指标字典，递归地将每一层的键加上 'val_' 前缀，并用下划线连接每一层的键。
        对每个指标调用 `compute()` 方法，并返回新的字典。

        Args:
            metrics (dict): 包含指标的字典。
            prefix (str): 前缀，默认为 'val'。

        Returns:
            dict: 更新后的字典，每一层的键都加上了 'val_' 前缀，并按要求进行了层级连接。
        """
        final_metrics = {}
        for key, value in metrics.items():
            new_key = f'{prefix}_{key}'
            if isinstance(value, dict):
                final_metrics.update(self.compute_metrics(value, prefix=new_key))
            else:
                final_metrics[new_key] = value.compute()
        return final_metrics


    def _validate(self, model, data_loader, step) -> dict[str, any]:
        self.reset_metrics()
        with torch.no_grad():
            with tqdm(
                total=len(data_loader),
                desc=f"Validate Step #{step + 1}",
                disable=not is_master(),
                file=sys.stdout,
            ) as t:
                for samples in data_loader:
                    # 获取数据
                    images = samples.get("image", None)
                    labels = samples.get("mask", None)


                    # 数据转到 GPU
                    images = images.cuda()

                    # 模型输入
                    model_input = {"image": images}
                    output = model(model_input)

                    # 模型可能返回多个输出时进行解包
                    if isinstance(output, dict):
                        main_output = output["main"]
                    else:
                        main_output = output

                    # 计算主任务的损失和指标
                    main_loss = self.test_criterion(main_output, labels)
                    predictions = torch.argmax(torch.softmax(main_output, dim=1), dim=1)
                    self.metrics["main"]["loss"].update(main_loss.item(), images.shape[0])
                    self.metrics["main"]["iou"].update(predictions, labels)
                    self.metrics["main"]["precision"].update(predictions, labels)
                    self.metrics["main"]["recall"].update(predictions, labels)
                    self.metrics["main"]["f1"].update(predictions, labels)


                    t.set_postfix(
                        {
                            "loss": self.metrics["main"]["loss"].compute().item() if "loss" in self.metrics["main"] else None,
                            #"#samples": self.metrics["main"]["loss"].get_count() if "loss" in self.metrics["main"] else None,
                            "bs": images.shape[0],
                        }
                    )
                    t.update()

        # 返回结果字典
        final_metrics = self.compute_metrics(self.metrics,"val")
        return final_metrics
    
    def before_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        """
        在训练步骤前根据配置的概率交换两张图片。如果 swap_probability 为 None，则不进行交换。

        Args:
            feed_dict (dict): 包含 "image", "t2_image", "mask", "t1_mask", 和 "t2_mask" 的输入字典。

        Returns:
            dict: 更新后的输入字典，可能已交换 image 和 t2_image。
        """
        # 获取图片和标签
        images = feed_dict.get("image", None)
        labels = feed_dict.get("mask", None)
        


        # 从 run_config 获取交换概率
        #swap_probability = getattr(self.run_config, "swap_probability", None)

        # 如果 swap_probability 不为 None，根据概率进行图片交换
        #if swap_probability is not None and torch.rand(1).item() < swap_probability:

        return {
            "image": images,
            "mask": labels,
        }
    def _run_step(self, feed_dict: dict[str, any]) -> dict[str, any]:
        """
        执行训练步骤，计算主任务和可能的 t1_mask、t2_mask 的损失、IoU 和 F1 Score。

        Args:
            feed_dict (dict[str, any]): 输入数据字典，包括 'data'、'label' 和可能的 't1_mask'、't2_mask'。

        Returns:
            dict[str, any]: 包含损失、IoU 和 F1 Score 的结果字典。
        """

        # 提取输入数据
        images = feed_dict["image"].cuda()
        labels = feed_dict.get("mask", None)
        labels = labels.long().cuda() if labels is not None else None


        model_input = {"image": images}
        # 检查是否需要使用 MESA
        if self.run_config.mesa is not None and self.run_config.mesa["thresh"] <= self.run_config.progress:
            ema_model = self.ema.shadows
            with torch.inference_mode():
                ema_output = ema_model(model_input).detach()
                if isinstance(ema_output, dict):
                    ema_main_output = ema_output["main"]
                else:
                    ema_main_output = ema_output
        else:
            ema_main_output = None

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                
            output = self.model(model_input)
            if isinstance(output, dict):
                main_output = output["main"]
            else:
                main_output = output
            # 主任务损失、IoU 和 F1 Score
            if main_output is not None and labels is not None:
                main_loss = self.train_criterion(main_output, labels)
                predictions = torch.argmax(torch.softmax(main_output, dim=1), dim=1)
                self.metrics["main"]["loss"].update(main_loss.item(), images.shape[0])
                self.metrics["main"]["iou"].update(predictions, labels)
                self.metrics["main"]["precision"].update(predictions, labels)
                self.metrics["main"]["recall"].update(predictions, labels)
                self.metrics["main"]["f1"].update(predictions, labels)
            else:
                main_loss=None

            if self.run_config.mesa is not None:
                if ema_main_output is not None:
                    mesa_loss = self.ema_loss(main_output, ema_main_output)
                self.metrics["mesa"]["loss"].update(mesa_loss.item(), main_output.shape[0])
            else:
                mesa_loss=None
            loss=0.0
            if main_loss is not None:
                loss += main_loss
            if mesa_loss is not None:
                loss += mesa_loss
        if not math.isfinite(loss.item()): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss.item()))
            assert math.isfinite(loss.item())
        self.scaler.scale(loss).backward()



    def _step_interval_training(self, epoch: int, total: int):
        self.reset_metrics()
        self.optimizer.zero_grad()
        print(f"Training data provider: {self.data_provider.train}")

        with tqdm(
            total=total,
            desc="Train Step #{}".format(epoch + 1),
            disable=not is_master(),
            file=sys.stdout,
        ) as t:
            for step_count, samples in enumerate(self.data_provider.train):
                # preprocessing
                feed_dict = self.before_step(samples)
                # clear gradient
                self.optimizer.zero_grad()
                # forward & backward
                self._run_step(feed_dict)
                # update: optimizer, lr_scheduler
                self.after_step()

                # 每log_interval步记录一次
                if (step_count + 1) % self.run_config.log_interval == 0:
                    # tqdm
                    postfix_dict = {
                        "loss": self.metrics["main"]["loss"].compute().item() if "loss" in self.metrics["main"] else None,
                        "bs": samples["image"].shape[0],
                        "lr": list_join(
                            sorted(set([group["lr"] for group in self.optimizer.param_groups])),
                            "#",
                            "%.1E",
                        ),
                        "progress": self.run_config.progress,
                    }
                    log_metrics = self.compute_metrics(self.metrics,"train")
                    if self.logger:
                        self.logger.log_metrics(metrics=log_metrics, step=epoch + step_count+1)


                    t.set_postfix(postfix_dict)
                    self.reset_metrics()
                t.update()
                if step_count >= total:
                    t.close()
                    break  # 只处理指定数量的批次


    def train(self, trials=0, save_freq=1) -> None:
        for epoch in range(self.start_epoch, self.run_config.n_steps, self.run_config.eval_interval):
            is_best = False
            # Training step
            self._step_interval_training(epoch, self.run_config.eval_interval)


            # Evaluate the model only every `eval_interval` steps
            if not self.run_config.train_only and (epoch + self.run_config.eval_interval) % self.run_config.eval_interval == 0:
                val_info_dict = self.validate(epoch=epoch + self.run_config.eval_interval)
                if self.logger:
                    self.logger.log_metrics(metrics=val_info_dict, step=epoch + self.run_config.eval_interval)

                # Compute average F1 score for validation
                avg_f1 = val_info_dict["val_main_f1"].item()
                is_best = avg_f1 > self.best_val
                self.best_val = max(avg_f1, self.best_val)

                # Auto restart condition based on abnormal accuracy drop
                if self.auto_restart_thresh is not None:
                    if self.best_val - avg_f1 > self.auto_restart_thresh:
                        self.write_log(f"Abnormal accuracy drop: {self.best_val} -> {avg_f1}")
                        self.load_model(os.path.join(self.checkpoint_path, "model_best.pt"))
                        return self.train(trials + 1, save_freq)



            # Save the model at specified intervals
            if (epoch + self.run_config.eval_interval) % save_freq == 0 or (is_best and self.run_config.progress > 0.8):
                self.save_model(
                        only_state_dict=False,
                        step=epoch + self.run_config.eval_interval,
                        model_name="model_best.pt" if is_best else "checkpoint.pt",
                    )