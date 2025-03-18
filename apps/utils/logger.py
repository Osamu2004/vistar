import os
import sys
import wandb
from typing import Optional, Dict

import os 
import sys
import wandb
from typing import Optional, Dict

class WandbLogger:
    # 定义一个类变量来存储当前实例
    _current_instance = None

    def __init__(self, project: str, config: Optional[Dict] = None, 
                 notes: Optional[str] = None, save_dir: Optional[str] = None, 
                 name: Optional[str] = None, offline: bool = False):
        """
        初始化 WandbLogger 类，用于记录训练日志。

        Args:
            project_name (str): wandb 项目的名称。
            config (dict, optional): 训练配置参数字典，将记录在 wandb 项目中。
            notes (str, optional): 在启动项目时附加的说明。
            save_dir (str, optional): 指定 wandb 的保存路径。如果没有提供，默认为当前工作目录。
            name (str, optional): 项目的名称。
            offline (bool): 是否以离线模式运行 wandb。
        """
        self.project_name = project
        self.config = config if config else {}
        self.notes = notes
        self.save_dir = save_dir if save_dir else os.getcwd()  # 默认为当前工作目录
        self.name = name
        self.offline = offline
        
        # 在初始化时将当前实例存储到类变量中
        WandbLogger._current_instance = self
        
        # 初始化 wandb
        self.init_wandb()

        # 设置全局异常捕获函数
        sys.excepthook = self.handle_exception

    def init_wandb(self):
        """优化后的初始化方法"""
        wandb_mode = "offline" if self.offline else None
        wandb.init(
            project=self.project_name,
            config=self.config,
            notes=self.notes,
            resume='allow' if self.offline else 'auto',  # 离线模式调整恢复策略
            dir=self.save_dir,
            name=self.name,
            mode=wandb_mode,  # 核心离线控制
            settings=wandb.Settings(
                start_method="thread",
                _disable_meta=True,  # 禁用元数据收集
            )
        )
        # 离线模式提示
        if self.offline:
            print(f"Wandb运行于离线模式，日志保存在: {self.save_dir}/wandb/")

    @classmethod
    def get_current_instance(cls):
        """
        返回当前 WandbLogger 实例。
        
        Returns:
            WandbLogger: 当前的 WandbLogger 实例。
        """
        return cls._current_instance

    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        记录一组指标数据到 wandb，并在 metrics 中增加一个 step 键。

        Args:
            metrics (dict): 需要记录的指标，如 loss, accuracy 等。
            step (int, optional): 当前的训练步数，若为 None 则不添加。
        """
        if step is not None:
            metrics['step'] = step  # 将 step 添加到 metrics 字典中
        
        # 记录到 wandb
        wandb.log(metrics)

    def log_artifact(self, artifact_name: str, artifact_type: str, artifact_data: str):
        """
        上传数据到 wandb 的 artifacts。

        Args:
            artifact_name (str): artifact 的名称。
            artifact_type (str): artifact 的类型（如 'model', 'dataset' 等）。
            artifact_data (str): 上传的文件路径。
        """
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(artifact_data)
        wandb.log({artifact_name: artifact})

    def finish(self):
        """
        结束当前的 wandb 记录。
        """
        try:
            wandb.finish()
        except Exception as e:
            print(f"Error finishing wandb: {e}")

    def handle_exception(self, exc_type, exc_value, exc_tb):
        """
        捕获未处理的异常并结束 wandb 会话。

        Args:
            exc_type: 异常类型。
            exc_value: 异常值。
            exc_tb: 异常的 traceback。
        """
        print(f"Uncaught exception: {exc_value}")
        
        try:
            if wandb.run and not self.offline:  # 离线模式不记录错误到云端
                wandb.log({"error": str(exc_value)})
                wandb.run.finish(exit_code=1)
        except Exception as e:
            print(f"Error handling exception: {e}")
        
        self.finish()
        sys.__excepthook__(exc_type, exc_value, exc_tb)
