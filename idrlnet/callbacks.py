"""Basic Callback classes"""

import os
import pathlib
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from idrlnet.receivers import Receiver, Signal
from idrlnet.variable import Variables

__all__ = ["GradientReceiver", "SummaryReceiver", "HandleResultReceiver"]


class GradientReceiver(Receiver):
    """Register the receiver to monitor gradient norm on the Tensorboard."""

    def receive_notify(self, solver: "Solver", message):  # noqa
        if not (Signal.TRAIN_PIPE_END in message):
            return
        for netnode in solver.netnodes:
            if not netnode.require_no_grad:
                model = netnode.net
                total_norm = 0
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)
                assert isinstance(solver.receivers[0], SummaryWriter)
                solver.summary_receiver.add_scalar(
                    "gradient/total_norm", total_norm, solver.global_step
                )


class SummaryReceiver(SummaryWriter, Receiver):
    """The receiver will be automatically registered to control the Tensorboard."""

    def __init__(self, *args, **kwargs):
        SummaryWriter.__init__(self, *args, **kwargs)

    def receive_notify(self, solver: "Solver", message: Dict):  # noqa
        if Signal.AFTER_COMPUTE_LOSS in message.keys():
            loss_component = message[Signal.AFTER_COMPUTE_LOSS]
            self.add_scalars("loss_overview", loss_component, solver.global_step)
            for key, value in loss_component.items():
                self.add_scalar(f"loss_component/{key}", value, solver.global_step)
        if Signal.TRAIN_PIPE_END in message.keys():
            for i, optimizer in enumerate(solver.optimizers):
                self.add_scalar(
                    f"optimizer/lr_{i}",
                    optimizer.param_groups[0]["lr"],
                    solver.global_step,
                )


class HandleResultReceiver(Receiver):
    """The receiver will be automatically registered to save results on training domains."""

    def __init__(self, result_dir):
        self.result_dir = result_dir

    def receive_notify(self, solver: "Solver", message: Dict):  # noqa
        if Signal.SOLVE_END in message.keys():
            samples = solver.sample_variables_from_domains()
            in_var, _, lambda_out = solver.generate_in_out_dict(samples)
            pred_out_sample = solver.forward_through_all_graph(
                in_var, solver.outvar_dict_index
            )
            diff_out_sample = {key: Variables() for key in pred_out_sample}
            results_path = pathlib.Path(self.result_dir)
            results_path.mkdir(exist_ok=True, parents=True)
            for key in samples:
                for _key in samples[key]:
                    if _key not in pred_out_sample[key].keys():
                        pred_out_sample[key][_key] = samples[key][_key]
                        diff_out_sample[key][_key] = samples[key][_key]
                    else:
                        diff_out_sample[key][_key] = (
                            pred_out_sample[key][_key] - samples[key][_key]
                        )
                samples[key].save(
                    os.path.join(results_path, f"{key}_true"), ["vtu", "np", "csv"]
                )
                pred_out_sample[key].save(
                    os.path.join(results_path, f"{key}_pred"), ["vtu", "np", "csv"]
                )
                diff_out_sample[key].save(
                    os.path.join(results_path, f"{key}_diff"), ["vtu", "np", "csv"]
                )
