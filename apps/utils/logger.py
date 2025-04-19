import os
import sys
from typing import Optional, Dict
import wandb


import os
import logging
from logging import FileHandler
from typing import Optional, Dict

class BaseLogger:
    """
    基础日志记录类，提供日志记录的基本接口，供其他日志记录类继承和扩展。
    """
    # 定义一个类变量来存储当前实例
    _current_instance = None

    def __init__(self, save_dir: Optional[str] = None, name: Optional[str] = None, log_filename="initialization.log"):
        """
        初始化基础日志记录类，设置保存目录和日志名称。
        
        Args:
            save_dir (str, optional): 日志保存目录，默认为当前工作目录。
            name (str, optional): 日志名称。
            log_filename (str, optional): 日志文件名称，默认为 'initialization.log'。
        """
        self.save_dir = save_dir if save_dir else os.getcwd()  # 默认为当前工作目录
        self.name = name
        self.log_filename = log_filename
        self.handlers = []

        # Create FileHandler for logging
        self._create_file_handler()

    def _create_file_handler(self):
        """Create a file handler for logging."""
        log_file_path = os.path.join(self.save_dir, self.log_filename)
        file_handler = FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # Formatter for logging
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the file handler to the list of handlers
        self.handlers.append(file_handler)

    @classmethod
    def get_current_instance(cls):
        """
        获取当前创建的日志实例，如果没有实例，创建并返回一个默认实例。
        """
        if cls._current_instance is None:
            cls._current_instance = BaseLogger()
        return cls._current_instance

    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """记录一组指标数据到日志。"""
        raise NotImplementedError("This method should be implemented by subclass.")

    def finish(self):
        """结束当前的日志记录。"""
        raise NotImplementedError("This method should be implemented by subclass.")




class WandbLogger(BaseLogger):
    """
    WandbLogger继承自BaseLogger，提供特定于wandb的日志记录功能。
    """



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
        super().__init__(save_dir, name)
        self.project_name = project
        self.config = config if config else {}
        self.notes = notes
        self.offline = offline

        # 在初始化时将当前实例存储到类变量中
        WandbLogger._current_instance = self

        # 初始化 wandb
        self.init_wandb()

        # 设置全局异常捕获函数
        sys.excepthook = self.handle_exception

    def init_wandb(self):
        """初始化wandb"""
        wandb_mode = "offline" if self.offline else None
        wandb.init(
            project=self.project_name,
            config=self.config,
            notes=self.notes,
            resume='allow' if self.offline else 'auto',
            dir=self.save_dir,
            name=self.name,
            mode=wandb_mode,
            settings=wandb.Settings(
                start_method="thread",
                _disable_meta=True,
            )
        )
        # 离线模式提示
        if self.offline:
            print(f"Wandb运行于离线模式，日志保存在: {self.save_dir}/wandb/")



    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        记录一组指标数据到wandb，并在metrics中增加一个step键。
        
        Args:
            metrics (dict): 需要记录的指标，如 loss, accuracy 等。
            step (int, optional): 当前的训练步数，若为 None 则不添加。
        """
        if step is not None:
            metrics['step'] = step
        
        # 记录到wandb
        wandb.log(metrics)

    def log_artifact(self, artifact_name: str, artifact_type: str, artifact_data: str):
        """
        上传数据到wandb的artifacts。
        
        Args:
            artifact_name (str): artifact的名称。
            artifact_type (str): artifact的类型（如'model', 'dataset'等）。
            artifact_data (str): 上传的文件路径。
        """
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(artifact_data)
        wandb.log({artifact_name: artifact})

    def finish(self):
        """
        结束当前的wandb记录。
        """
        try:
            wandb.finish()
        except Exception as e:
            print(f"Error finishing wandb: {e}")

    def handle_exception(self, exc_type, exc_value, exc_tb):
        """
        捕获未处理的异常并结束wandb会话。
        
        Args:
            exc_type: 异常类型。
            exc_value: 异常值。
            exc_tb: 异常的traceback。
        """
        print(f"Uncaught exception: {exc_value}")
        
        try:
            if wandb.run and not self.offline:
                wandb.log({"error": str(exc_value)})
                wandb.run.finish(exit_code=1)
        except Exception as e:
            print(f"Error handling exception: {e}")
        
        self.finish()
        sys.__excepthook__(exc_type, exc_value, exc_tb)

