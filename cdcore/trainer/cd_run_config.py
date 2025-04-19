from apps.trainer.run_config import RunStepConfig

__all__ = ["CDRunConfig"]


class CDRunConfig(RunStepConfig):
    @property
    def none_allowed(self):
        return [] + super().none_allowed