"""Define Optimizers and LR schedulers"""

import abc
import torch
import inspect
import math
from typing import Dict

__all__ = ["get_available_class", "Optimizable"]


def get_available_class(module, class_name) -> Dict[str, type]:
    """Search specified subclasses of the given class in module.

    :param module: The module name
    :type module: module
    :param class_name: the parent class
    :type class_name: type
    :return: A dict mapping from subclass.name to subclass
    :rtype: Dict[str, type]
    """
    return dict(
        filter(
            lambda x: inspect.isclass(x[1])
            and issubclass(x[1], class_name)
            and (not x[1] == class_name),
            inspect.getmembers(module),
        )
    )


class Optimizable(metaclass=abc.ABCMeta):
    """An abstract class for organizing optimization related configuration and operations.
    The interface is implemented by solver.Solver
    """

    OPTIMIZER_MAP = get_available_class(
        module=torch.optim, class_name=torch.optim.Optimizer
    )
    SCHEDULE_MAP = get_available_class(
        module=torch.optim.lr_scheduler,
        class_name=torch.optim.lr_scheduler._LRScheduler,
    )

    @property
    def optimizers(self):
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers):
        self._optimizers = optimizers

    @property
    def schedulers(self):
        return self._schedulers

    @schedulers.setter
    def schedulers(self, schedulers):
        self._schedulers = schedulers

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    def parse_configure(self, **kwargs):
        self.parse_optimizer(**kwargs)
        self.parse_lr_schedule(**kwargs)
        self.configure_optimizers()

    def parse_optimizer(self, **kwargs):
        default_config = dict(optimizer="Adam", lr=1e-3)
        default_config.update(kwargs.get("opt_config", {}))
        self.optimizer_config = default_config

    def parse_lr_schedule(self, **kwargs):
        default_config = dict(
            scheduler="ExponentialLR", gamma=math.pow(0.95, 0.001), last_epoch=-1
        )
        default_config.update(kwargs.get("schedule_config", {}))
        self.schedule_config = default_config

    def __str__(self):
        if "optimizer_config" in self.__dict__:
            opt_str = str(self.optimizer_config)
        else:
            opt_str = str("optimizer is empty...")

        if "schedule_config" in self.__dict__:
            schedule_str = str(self.schedule_config)
        else:
            schedule_str = str("scheduler is empty...")
        return "\n".join([opt_str, schedule_str])
