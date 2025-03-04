
import argparse
import math
import os

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from apps.utils.model import (
    build_kwargs_from_config,
)
from apps.builder import make_dataprovider
from cdcore.registry import register_cd_all
from apps.utils.misc import  parse_unknown_args
from apps import setup
from cdcore.cd_model_zoo import create_cd_model
from torchmetrics import MeanMetric, JaccardIndex, Precision, Recall, F1Score

parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file")


def initialize_metrics(data_provider):

    num_classes = data_provider.main_classes
        # 初始化主要评价指标
    metrics = {
            "main": {
                "iou": JaccardIndex(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes).cuda(),
                "precision": Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
                "recall": Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
                "f1": F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
            }
        }
    if "t1_mask" in data_provider.samples:
        metrics["t1_mask"] = {
                "iou": JaccardIndex(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes).cuda(),
                "precision": Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
                "recall": Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
                "f1": F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
            }

        # 如果 t2_mask 存在，则初始化 t2_mask 的指标
        if "t2_mask" in data_provider.samples:
            metrics["t2_mask"] = {
                "iou": JaccardIndex(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes),
                "precision": Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
                "recall": Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
                "f1": F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes, average="macro").cuda(),
            }

    return metrics


def main():
    register_cd_all()

    args = parser.parse_args()

    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)
    config["data_provider"]["only_test"] = True
    data_provider = setup.setup_data_provider(config, is_distributed=False)
    model = create_cd_model(config["net_config"]["name"],pretrained = True,dataset = config['data_provider'].get('type'),
                            weight_url=r"E:\zzy\mmchange0.2\run_dir\checkpoint\model_best.pt",
                            )
    #model = torch.nn.DataParallel(model).cuda()
    print(data_provider.test)
    model.cuda().eval()
    for name, param in model.named_parameters():
        print(name, param.mean())  
    metrics = initialize_metrics(data_provider)
    for key, metric_dict in metrics.items():
        for metric_name, metric in metric_dict.items():
            metric.reset()  

    with torch.no_grad():
        with tqdm(
                total=len(data_provider.test),
                desc=f"Validate Step #",
            ) as t:
            for samples in data_provider.test:
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

                predictions = torch.argmax(torch.softmax(main_output, dim=1), dim=1)
                metrics["main"]["iou"].update(predictions, labels)
                metrics["main"]["precision"].update(predictions, labels)
                metrics["main"]["recall"].update(predictions, labels)
                metrics["main"]["f1"].update(predictions, labels)

                    # 如果 t1_mask 存在，则计算 t1_mask 的损失和指标
                if t1_output is not None and t1_mask is not None:
                    predictions_t1 = torch.argmax(torch.softmax(t1_output, dim=1), dim=1)
                    metrics["t1_mask"]["iou"].update(predictions_t1, t1_mask)
                    metrics["t1_mask"]["precision"].update(predictions_t1, t1_mask)
                    metrics["t1_mask"]["recall"].update(predictions_t1, t1_mask)
                    metrics["t1_mask"]["f1"].update(predictions_t1, t1_mask)

                    # 如果 t2_mask 存在，则计算 t2_mask 的损失和指标
                if t2_output is not None and t2_mask is not None:
                    predictions_t2 = torch.argmax(torch.softmax(t2_output, dim=1), dim=1)
                    metrics["t2_mask"]["iou"].update(predictions_t2, t2_mask)
                    metrics["t2_mask"]["precision"].update(predictions_t2, t2_mask)
                    metrics["t2_mask"]["recall"].update(predictions_t2, t2_mask)
                    metrics["t2_mask"]["f1"].update(predictions_t2, t2_mask)

                t.set_postfix(
                        {
                            "loss": metrics["main"]["loss"].compute().item() if "loss" in metrics["main"] else None,
                            "iou": metrics["main"]["iou"].compute().item() if "iou" in metrics["main"] else None,
                            "precision": metrics["main"]["precision"].compute().item() if "precision" in metrics["main"] else None,
                            "recall": metrics["main"]["recall"].compute().item() if "recall" in metrics["main"] else None,
                            "f1": metrics["main"]["f1"].compute().item() if "f1" in metrics["main"] else None,
                            "t1_loss": metrics["t1_mask"]["loss"].compute().item() if "t1_mask" in metrics else None,
                            "t2_loss": metrics["t2_mask"]["loss"].compute().item() if "t2_mask" in metrics else None,
                            #"#samples": self.metrics["main"]["loss"].get_count() if "loss" in self.metrics["main"] else None,
                            "bs": images.shape[0],
                        }
                    )
                t.update()

        # 返回结果字典
        final_metrics = {}
        for key, metric_dict in metrics.items():
            final_metrics[key] = {name: metric.compute() for name, metric in metric_dict.items()}
        print(final_metrics)

        return final_metrics




if __name__ == "__main__":
    main()