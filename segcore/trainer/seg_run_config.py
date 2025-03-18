from apps.trainer.run_config import RunStepConfig

__all__ = ["SegRunConfig"]


class SegRunConfig(RunStepConfig):
    @property
    def none_allowed(self):
        return [] + super().none_allowed