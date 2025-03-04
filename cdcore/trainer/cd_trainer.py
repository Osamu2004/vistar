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
from apps.utils.metric import calculate_iou,calculate_precision,calculate_recall,calculate_f1_score
from apps.builder import make_loss
from apps.utils.model import list_join, list_mean
from torchmetrics import MeanMetric, JaccardIndex, Precision, Recall, F1Score
from apps.builder import make_loss
__all__ = ["CDTrainer"]
import pdb

class CDTrainer(StepTrainer):
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
        ignore_index = getattr(self.data_provider, 'ignore_index', None)

        # 初始化主要评价指标
        metrics = {
            "main": {
                "loss": MeanMetric().cuda(),  # 平均损失
                "iou": JaccardIndex(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes,ignore_index=ignore_index).cuda(),
                "precision": Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
                "recall": Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
                "f1": F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
            }
        }
        if "t1_mask" in self.data_provider.samples:
            metrics["t1_mask"] = {
                "loss": MeanMetric().cuda(),
                "iou": JaccardIndex(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes,ignore_index=ignore_index).cuda(),
                "precision": Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
                "recall": Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
                "f1": F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
            }

        # 如果 t2_mask 存在，则初始化 t2_mask 的指标
        if "t2_mask" in self.data_provider.samples:
            metrics["t2_mask"] = {
                "loss": MeanMetric().cuda(),
                "iou": JaccardIndex(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes,ignore_index=ignore_index),
                "precision": Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
                "recall": Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
                "f1": F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro",ignore_index=ignore_index).cuda(),
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
                    t2_images = samples.get("t2_image", None)
                    labels = samples.get("mask", None)
                    t1_mask = samples.get("t1_mask", None)
                    t2_mask = samples.get("t2_mask", None)

                    if images is None or labels is None or t2_images is None:
                        raise ValueError("The provided dataloader must return a dictionary with at least 'image', 't2_image', and 'mask' keys.")

                    # 数据转到 GPU
                    images = images.cuda()
                    t2_images = t2_images.cuda()
                    labels = labels.long().cuda()
                    if t1_mask is not None:
                        t1_mask = t1_mask.long().cuda()
                    if t2_mask is not None:
                        t2_mask = t2_mask.long().cuda()

                    # 模型输入
                    model_input = {"image": images, "t2_image": t2_images}
                    output = model(model_input)

                    # 模型可能返回多个输出时进行解包
                    if isinstance(output, dict):
                        main_output = output["main"]
                        t1_output = output.get("t1_mask", None)
                        t2_output = output.get("t2_mask", None)
                    else:
                        main_output = output
                        t1_output = t2_output = None

                    # 计算主任务的损失和指标
                    main_loss = self.test_criterion(main_output, labels)
                    predictions = torch.argmax(torch.softmax(main_output, dim=1), dim=1)
                    self.metrics["main"]["loss"].update(main_loss.item(), images.shape[0])
                    self.metrics["main"]["iou"].update(predictions, labels)
                    self.metrics["main"]["precision"].update(predictions, labels)
                    self.metrics["main"]["recall"].update(predictions, labels)
                    self.metrics["main"]["f1"].update(predictions, labels)

                    # 如果 t1_mask 存在，则计算 t1_mask 的损失和指标
                    if t1_output is not None and t1_mask is not None:
                        t1_loss = self.test_criterion(t1_output, t1_mask)
                        predictions_t1 = torch.argmax(torch.softmax(t1_output, dim=1), dim=1)
                        self.metrics["t1_mask"]["loss"].update(t1_loss.item(), t1_mask.shape[0])
                        self.metrics["t1_mask"]["iou"].update(predictions_t1, t1_mask)
                        self.metrics["t1_mask"]["precision"].update(predictions_t1, t1_mask)
                        self.metrics["t1_mask"]["recall"].update(predictions_t1, t1_mask)
                        self.metrics["t1_mask"]["f1"].update(predictions_t1, t1_mask)

                    # 如果 t2_mask 存在，则计算 t2_mask 的损失和指标
                    if t2_output is not None and t2_mask is not None:
                        t2_loss = self.test_criterion(t2_output, t2_mask)
                        predictions_t2 = torch.argmax(torch.softmax(t2_output, dim=1), dim=1)
                        self.metrics["t2_mask"]["loss"].update(t2_loss.item(), t2_mask.shape[0])
                        self.metrics["t2_mask"]["iou"].update(predictions_t2, t2_mask)
                        self.metrics["t2_mask"]["precision"].update(predictions_t2, t2_mask)
                        self.metrics["t2_mask"]["recall"].update(predictions_t2, t2_mask)
                        self.metrics["t2_mask"]["f1"].update(predictions_t2, t2_mask)

                    t.set_postfix(
                        {
                            "loss": self.metrics["main"]["loss"].compute().item() if "loss" in self.metrics["main"] else None,
                            "t1_loss": self.metrics["t1_mask"]["loss"].compute().item() if "t1_mask" in self.metrics else None,
                            "t2_loss": self.metrics["t2_mask"]["loss"].compute().item() if "t2_mask" in self.metrics else None,
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
        t2_images = feed_dict.get("t2_image", None)
        labels = feed_dict.get("mask", None)
        t1_mask = feed_dict.get("t1_mask", None)
        t2_mask = feed_dict.get("t2_mask", None)
        

        if images is None or t2_images is None:
            raise ValueError("The provided feed_dict must contain both 'image' and 't2_image' keys.")

        # 从 run_config 获取交换概率
        #swap_probability = getattr(self.run_config, "swap_probability", None)

        # 如果 swap_probability 不为 None，根据概率进行图片交换
        #if swap_probability is not None and torch.rand(1).item() < swap_probability:
        if self.run_config.exchange and torch.rand(1).item() < 0.5:
            images, t2_images = t2_images, images  # 交换图片

        return {
            "image": images,
            "t2_image": t2_images,
            "mask": labels,
            "t1_mask": t1_mask,
            "t2_mask": t2_mask,
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
        t2_images = feed_dict["t2_image"].cuda()
        labels = feed_dict["mask"].long().cuda()
        t1_mask = feed_dict.get("t1_mask", None)
        t1_mask = t1_mask.long().cuda() if t1_mask is not None else None  # 保留 None

        t2_mask = feed_dict.get("t2_mask", None)
        t2_mask = t2_mask.long().cuda() if t2_mask is not None else None  # 保留 None
        model_input = {"image": images}
        model_input["t2_image"] = t2_images
        # 检查是否需要使用 MESA
        if self.run_config.mesa is not None and self.run_config.mesa["thresh"] <= self.run_config.progress:
            ema_model = self.ema.shadows
            with torch.inference_mode():
                ema_output = ema_model(model_input).detach()
                if isinstance(ema_output, dict):
                    ema_main_output = ema_output["main"]
                    ema_t1_output = ema_output.get("t1_mask", None)
                    ema_t2_output = ema_output.get("t2_mask", None)
                else:
                    ema_main_output = ema_output
                    ema_t1_output = ema_t2_output = None
        else:
            ema_main_output = ema_t1_output = ema_t2_output = None

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                
            output = self.model(model_input)
            if isinstance(output, dict):
                main_output = output["main"]
                t1_output = output.get("t1_mask", None)
                t2_output = output.get("t2_mask", None)
            else:
                main_output = output
                t1_output = t2_output = None
            # 主任务损失、IoU 和 F1 Score
            main_loss = self.train_criterion(main_output, labels)
            predictions = torch.argmax(torch.softmax(main_output, dim=1), dim=1)
            if ema_main_output is not None:
                mesa_main_loss = self.ema_loss(main_output, ema_main_output)
                main_loss = main_loss +  mesa_main_loss
            self.metrics["main"]["loss"].update(main_loss.item(), images.shape[0])
            self.metrics["main"]["iou"].update(predictions, labels)
            self.metrics["main"]["precision"].update(predictions, labels)
            self.metrics["main"]["recall"].update(predictions, labels)
            self.metrics["main"]["f1"].update(predictions, labels)

            # 如果存在 t1_mask，则计算相关损失、IoU 和 F1 Score
            if t1_output is not None and t1_mask is not None:
                t1_loss = self.train_criterion(t1_output, t1_mask)
                predictions_t1 = torch.argmax(torch.softmax(t1_output, dim=1), dim=1)

                if ema_t1_output is not None:
                    mesa_t1_loss = self.ema_loss(t1_output, ema_t1_output)
                    t1_loss = t1_loss +  mesa_t1_loss

                self.metrics["t1_mask"]["loss"].update(t1_loss.item(), t1_mask.shape[0])
                self.metrics["t1_mask"]["iou"].update(predictions_t1, t1_mask)
                self.metrics["t1_mask"]["precision"].update(predictions_t1, t1_mask)
                self.metrics["t1_mask"]["recall"].update(predictions_t1, t1_mask)
                self.metrics["t1_mask"]["f1"].update(predictions_t1, t1_mask)
            else:
                t1_loss = None

                # 如果存在 t2_mask，则计算相关损失、IoU 和 F1 Score
            if t2_output is not None and t2_mask is not None:
                t2_loss = self.train_criterion(t2_output, t2_mask)
                predictions_t2 = torch.argmax(torch.softmax(t2_output, dim=1), dim=1)

                if ema_t2_output is not None:
                    mesa_t2_loss = self.ema_loss(t2_output, ema_t2_output)
                    t2_loss = t2_loss +  mesa_t2_loss

                self.metrics["t2_mask"]["loss"].update(t2_loss.item(), t2_mask.shape[0])
                self.metrics["t2_mask"]["iou"].update(predictions_t2, t2_mask)
                self.metrics["t2_mask"]["precision"].update(predictions_t2, t2_mask)
                self.metrics["t2_mask"]["recall"].update(predictions_t2, t2_mask)
                self.metrics["t2_mask"]["f1"].update(predictions_t2, t2_mask)
            else:
                t2_loss = None
            loss = main_loss
            if t1_loss is not None:
                loss += t1_loss
            if t2_loss is not None:
                loss += t2_loss
        if not math.isfinite(loss.item()): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss.item()))
            assert math.isfinite(loss.item())
        self.scaler.scale(loss).backward()


    '''
    def _step_interval_training(self, epoch: int, total: int, log_interval: int) -> dict[str, any]:
        self.reset_metrics()
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
                if "t1_mask" in self.metrics:
                    postfix_dict["t1_loss"] = self.metrics["t1_mask"]["loss"].compute().item()
                if "t2_mask" in self.metrics:
                    postfix_dict["t2_loss"] = self.metrics["t2_mask"]["loss"].compute().item()
                t.set_postfix(postfix_dict)
                t.update()
                if step_count >= total:
                    t.close()
                    break  # 只处理指定数量的批次
        final_metrics = self.compute_metrics(self.metrics,"train")
        return final_metrics

    '''
    def _step_interval_training(self, epoch: int, total: int):
        self.reset_metrics()
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
                    if "t1_mask" in self.metrics:
                        postfix_dict["t1_loss"] = self.metrics["t1_mask"]["loss"].compute().item()
                    if "t2_mask" in self.metrics:
                        postfix_dict["t2_loss"] = self.metrics["t2_mask"]["loss"].compute().item()
                    log_metrics = self.compute_metrics(self.metrics,"train")
                    if self.logger:
                        self.logger.log_metrics(metrics=log_metrics, step=epoch + step_count+1)


                    t.set_postfix(postfix_dict)
                    self.reset_metrics()
                t.update()
                if step_count >= total:
                    t.close()
                    break  # 只处理指定数量的批次

    '''
    def train(self, trials=0, save_freq=1) -> None:
        for epoch in range(self.start_epoch, self.run_config.n_steps,self.run_config.log_interval):
            train_info_dict = self._step_interval_training(epoch,self.run_config.log_interval)
            if self.logger:
                self.logger.log_metrics(metrics = train_info_dict,step = epoch+self.run_config.log_interval)
            # eval
            val_info_dict = self.validate(epoch=epoch+self.run_config.log_interval)
            if self.logger:
                self.logger.log_metrics(metrics = val_info_dict,step = epoch+self.run_config.log_interval)

            avg_f1 = list_mean([info_dict["f1"] for info_dict in val_info_dict.values()])
            is_best = avg_f1 > self.best_val
            self.best_val = max(avg_f1, self.best_val)

            if self.auto_restart_thresh is not None:
                if self.best_val - avg_f1 > self.auto_restart_thresh:
                    self.write_log(f"Abnormal accuracy drop: {self.best_val} -> {avg_f1}")
                    self.load_model(os.path.join(self.checkpoint_path, "model_best.pt"))
                    return self.train(trials + 1, save_freq)

            # log
            val_log = self.run_config.step_format(epoch+self.run_config.log_interval)
            val_log += f"\tval_f1={avg_f1:.4f}({self.best_val:.4f})"
            val_log += "\tVal("
            for key in list(val_info_dict.values())[0]:
                if key == "f1":
                    continue
                val_log += f"{key}={list_mean([info_dict[key] for info_dict in val_info_dict.values()]):.4f},"
            val_log += ")\tTrain("
            for key, val in train_info_dict.items():
                val_log += f"{key}={val:.4f},"
            val_log += (
                f'lr={list_join(sorted(set([group["lr"] for group in self.optimizer.param_groups])), "#", "%.2E")})'
            )
            self.write_log(val_log, prefix="valid", print_log=False)

            # save model
            
            if (epoch+self.run_config.log_interval) % save_freq == 0 or (is_best and self.run_config.progress > 0.8):
                self.save_model(
                    only_state_dict=False,
                    step=epoch+self.run_config.log_interval,
                    model_name="model_best.pt" if is_best else "checkpoint.pt",
                )
    
    '''

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