"""Concrete predefined callbacks"""

import abc
from enum import Enum
from typing import Dict, List


class Signal(Enum):
    REGISTER = "signal_register"
    SOLVE_START = "signal_solve_start"
    TRAIN_PIPE_START = "signal_train_pipe_start"
    BEFORE_COMPUTE_LOSS = "before_compute_loss"
    AFTER_COMPUTE_LOSS = "compute_loss"
    BEFORE_BACKWARD = "signal_before_backward"
    TRAIN_PIPE_END = "signal_train_pipe_end"
    SOLVE_END = "signal_solve_end"


class Receiver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def receive_notify(self, obj: object, message: Dict):
        raise NotImplementedError("Method receive_notify() not implemented!")


class Notifier:
    @property
    def receivers(self):
        return self._receivers

    @receivers.setter
    def receivers(self, receivers: List[Receiver]):
        self._receivers = receivers

    def notify(self, obj: object, message: Dict):
        for receiver in self.receivers[::-1]:
            receiver.receive_notify(obj, message)

    def register_receiver(self, receiver: Receiver):
        self.receivers.append(receiver)
        self.notify(self, message={Signal.REGISTER: receiver})
