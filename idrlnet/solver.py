"""Solver"""

from collections import ChainMap
import torch
import os
import pathlib
from typing import Dict, List, Union, Tuple, Optional, Callable
from idrlnet.callbacks import SummaryReceiver, HandleResultReceiver
from idrlnet.header import logger
from idrlnet.optim import Optimizable
from idrlnet.data import DataNode, SampleDomain
from idrlnet.net import NetNode
from idrlnet.receivers import Receiver, Notifier, Signal
from idrlnet.variable import Variables, DomainVariables
from idrlnet.graph import VertexTaskPipeline
import idrlnet

__all__ = ["Solver"]


class Solver(Notifier, Optimizable):
    """Instances of the Solver class integrate configurations and handle the computation
    operation during solving PINNs. One problem usually needs one instance to solve.

    :param sample_domains: A tuple of geometry domains used to sample points for training of PINNs.
    :type sample_domains: Tuple[DataNode, ...]
    :param netnodes: A list of neural networks. Trainable computation nodes.
    :type netnodes: List[NetNode]
    :param pdes: A list of partial differential equations. Similar to net nodes, they can evaluateinputs and output
                 results. But they are not trainable.
    :type pdes: Optional[List[PdeNode]]
    :param network_dir: The directory used to automatically load and store ckpt files
    :type network_dir: str
    :param summary_dir: The directory is used for store information about tensorboard. If it is not specified,
                        it will be assigned to network_dir by default.
    :type summary_dir: Optional[str]
    :param max_iter: Max iteration the solver would run.
    :type max_iter: int
    :param save_freq: Frequency of saving ckpt.
    :type save_freq: int
    :param print_freq: Frequency of printing loss.
    :type print_freq: int
    :param loading: By default, it is true. It will try to load ckpt and continue previous training stage.
    :type loading: bool
    :param init_network_dirs: A list of directories for loading pre-trained networks.
    :type init_network_dirs: List[str]
    :param opt_config: Configure one optimizer for all trainable parameters. It is a wrapper of `torch.optim.Optimizer`.
                       One can specify any subclasses of `torch.optim.Optimizer` by
                       expanding the args like:

                       -  `opt_config=dict(optimizer='Adam', lr=0.001)` **by default**.
                       -  `opt_config=dict(optimizer='SGD', lr=0.01, momentum=0.9)`
                       -  `opt_config=dict(optimizer='SparseAdam', lr=0.001, betas=(0.9, 0.999), eps=1e-08)`
                       Note that the opt is Case Sensitive.
    :type opt_config: Dict
    :param schedule_config: Configure one lr scheduler for the optimizer. It is a wrapper of

                            - `torch.optim.lr_scheduler._LRScheduler`. One can specify any subclasses of the class lke:
                            - `schedule_config=dict(scheduler='ExponentialLR', gamma=math.pow(0.95, 0.001))`
                            - `schedule_config=dict(scheduler='StepLR', step_size=30, gamma=0.1)`
                            Note that the scheduler is Case Sensitive.
    :type schedule_config: Dict
    :param result_dir: save the final training domain data. defaults to 'train_domain/results'
    :type result_dir: str
    :param kwargs:
    """

    def __init__(
        self,
        sample_domains: Tuple[Union[DataNode, SampleDomain], ...],
        netnodes: List[NetNode],
        pdes: Optional[List] = None,
        network_dir: str = "./network_dir",
        summary_dir: Optional[str] = None,
        max_iter: int = 1000,
        save_freq: int = 100,
        print_freq: int = 10,
        loading: bool = True,
        init_network_dirs: Optional[List[str]] = None,
        opt_config: Dict = None,
        schedule_config: Dict = None,
        result_dir="train_domain/results",
        **kwargs,
    ):

        self.network_dir: str = network_dir
        self.domain_losses = {domain.name: domain.loss_fn for domain in sample_domains}
        self.netnodes: List[NetNode] = netnodes
        if init_network_dirs:
            self.init_network_dirs = init_network_dirs
        else:
            self.init_network_dirs = []
        self.init_load()

        self.pdes: List = [] if pdes is None else pdes
        pathlib.Path(self.network_dir).mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.max_iter = max_iter
        self.save_freq = save_freq
        self.print_freq = print_freq
        try:
            self.parse_configure(
                **{
                    **({"opt_config": opt_config} if opt_config is not None else {}),
                    **(
                        {"schedule_config": schedule_config}
                        if schedule_config is not None
                        else {}
                    ),
                }
            )
        except Exception:
            logger.error("Optimizer configuration failed")
            raise

        if loading:
            try:
                self.load()
            except:
                pass
        self.sample_domains: Tuple[DataNode, ...] = sample_domains
        self.summary_dir = self.network_dir if summary_dir is None else summary_dir
        self.receivers: List[Receiver] = [
            SummaryReceiver(self.summary_dir),
            HandleResultReceiver(result_dir),
        ]

    @property
    def network_dir(self):
        return self._network_dir

    @network_dir.setter
    def network_dir(self, network_dir):
        self._network_dir = network_dir

    @property
    def sample_domains(self):
        return self._sample_domains

    @sample_domains.setter
    def sample_domains(self, sample_domains):
        self._sample_domains = sample_domains
        self._generate_dict_index()
        self.generate_computation_pipeline()

    @property
    def trainable_parameters(self) -> List[torch.nn.parameter.Parameter]:
        """Return trainable parameters in netnodes. Parameters in netnodes with ``is_reference=True``
        or ``fixed=True`` will not be returned.
        :return: A list of trainable parameters.
        :rtype: List[torch.nn.parameter.Parameter]
        """
        parameter_list = list(
            map(
                lambda _net_node: {"params": _net_node.net.parameters()},
                filter(
                    lambda _net_node: not _net_node.is_reference
                    and (not _net_node.fixed),
                    self.netnodes,
                ),
            )
        )
        if len(parameter_list) == 0:
            """To make sure successful initialization of optimizers."""
            parameter_list = [
                torch.nn.parameter.Parameter(
                    data=torch.Tensor([0.0]), requires_grad=True
                )
            ]
            logger.warning("No trainable parameters found!")
        return parameter_list

    @property
    def summary_receiver(self) -> SummaryReceiver:
        try:
            summary_receiver = self.receivers[0]
            assert isinstance(summary_receiver, SummaryReceiver)
        except IndexError:
            raise
        return summary_receiver

    def __str__(self):
        """return sovler information, it will return components recursively"""
        str_list = []
        str_list.append("nets: \n")
        str_list.append("".join([str(net) for net in self.netnodes]))
        str_list.append("domains: \n")
        str_list.append("".join([str(domain) for domain in self.sample_domains]))
        str_list.append("\n")
        str_list.append("optimizer config:\n")
        for i, _class in enumerate(type(self).mro()):
            if _class == Optimizable:
                str_list.append(super(type(self).mro()[i - 1], self).__str__())
        return "".join(str_list)

    def set_param_ranges(self, param_ranges: Dict):
        for domain in self.sample_domains:
            domain.sample_fn.param_ranges = param_ranges

    def set_domain_parameter(self, domain_name: str, parameter_dict: dict):
        domain = self.get_sample_domain(domain_name)
        for key, value in parameter_dict.items():
            domain.sample_fn.__dict__[key] = value

    def get_domain_parameter(self, domain_name: str, parameter: str):
        return self.get_sample_domain(domain_name).sample_fn.__dict__[parameter]

    def get_sample_domain(self, name: str) -> DataNode:
        for value in self.sample_domains:
            if value.name == name:
                return value
        raise KeyError(f"domain {name} not exist!")

    def generate_computation_pipeline(self):
        """Generate computation pipeline for all domains.
        The change of `self.sample_domains` will triger this method.
        """
        samples = self.sample_variables_from_domains()
        in_var, true_out, lambda_out = self.generate_in_out_dict(samples)
        self.vertex_pipelines = {}
        for domain_name, var in in_var.items():
            logger.info(f"Constructing computation graph for domain <{domain_name}>")
            self.vertex_pipelines[domain_name] = VertexTaskPipeline(
                self.netnodes + self.pdes, var, self.outvar_dict_index[domain_name]
            )
            self.vertex_pipelines[domain_name].display(
                os.path.join(self.network_dir, f"{domain_name}_{self.global_step}.png")
            )

    def forward_through_all_graph(
        self, invar_dict: DomainVariables, req_outvar_dict_index: Dict[str, List[str]]
    ) -> DomainVariables:
        outvar_dict = {}
        for (key, req_outvar_names) in req_outvar_dict_index.items():
            outvar_dict[key] = self.vertex_pipelines[key].forward_pipeline(
                invar_dict[key], req_outvar_names
            )
        return outvar_dict

    def append_sample_domain(self, datanode):
        self.sample_domains = self.sample_domains + (datanode,)

    def _generate_dict_index(self) -> None:
        self.invar_dict_index = {
            domain.name: domain.inputs for domain in self.sample_domains
        }
        self.outvar_dict_index = {
            domain.name: domain.outputs for domain in self.sample_domains
        }
        self.lambda_dict_index = {
            domain.name: domain.lambda_outputs for domain in self.sample_domains
        }

    def generate_in_out_dict(
        self, samples: DomainVariables
    ) -> Tuple[DomainVariables, DomainVariables, DomainVariables]:
        invar_dict = {}
        for domain, variable in samples.items():
            inner = {}
            for key, val in variable.items():
                if key in self.invar_dict_index[domain]:
                    inner[key] = val
            invar_dict[domain] = inner

        invar_dict = {
            domain: Variables(
                {
                    key: val
                    for key, val in variable.items()
                    if key in self.invar_dict_index[domain]
                }
            )
            for domain, variable in samples.items()
        }
        outvar_dict = {
            domain: Variables(
                {
                    key: val
                    for key, val in variable.items()
                    if key in self.outvar_dict_index[domain]
                }
            )
            for domain, variable in samples.items()
        }
        lambda_dict = {
            domain: Variables(
                {
                    key: val
                    for key, val in variable.items()
                    if key in self.lambda_dict_index[domain]
                }
            )
            for domain, variable in samples.items()
        }
        return invar_dict, outvar_dict, lambda_dict

    def solve(self):
        """After the solver instance is initialized, the method could be called to solve the entire problem."""
        self.notify(self, message={Signal.SOLVE_START: "default"})
        while self.global_step < self.max_iter:
            loss = self.train_pipe()
            if self.global_step % self.print_freq == 0:
                logger.info("Iteration: {}, Loss: {}".format(self.global_step, loss))
            if self.global_step % self.save_freq == 0:
                self.save()
        logger.info("Training Stage Ends")
        self.notify(self, message={Signal.SOLVE_END: "default"})

    def train_pipe(self):
        """Sample once; calculate the loss once; backward propagation once
        :return: None
        """
        self.notify(self, message={Signal.TRAIN_PIPE_START: "defaults"})
        for opt in self.optimizers:
            opt.zero_grad()
        samples = self.sample_variables_from_domains()
        in_var, true_out, lambda_out = self.generate_in_out_dict(samples)
        pred_out_sample = self.forward_through_all_graph(in_var, self.outvar_dict_index)
        try:
            loss = self.compute_loss(in_var, pred_out_sample, true_out, lambda_out)
        except RuntimeError:
            raise
        self.notify(self, message={Signal.BEFORE_BACKWARD: "defaults"})
        loss.backward()
        for opt in self.optimizers:
            opt.step()
        self.global_step += 1

        for scheduler in self.schedulers:
            scheduler.step(self.global_step)
        self.notify(self, message={Signal.TRAIN_PIPE_END: "defaults"})
        return loss

    def compute_loss(
        self,
        in_var: DomainVariables,
        pred_out_sample: DomainVariables,
        true_out: DomainVariables,
        lambda_out: DomainVariables,
    ) -> torch.Tensor:
        """Compute the total loss in one epoch."""
        diff = dict()
        for domain_name, domain_val in true_out.items():
            if len(domain_val) == 0:
                continue
            diff[domain_name] = (
                pred_out_sample[domain_name] - domain_val.to_torch_tensor_()
            )
            diff[domain_name].update(lambda_out[domain_name])
            diff[domain_name].update(area=in_var[domain_name]["area"])

        for domain, var in diff.items():
            lambda_diff = dict()
            for constraint, _ in var.items():
                if "lambda_" + constraint in in_var[domain].keys():
                    lambda_diff["lambda_" + constraint] = in_var[domain][
                        "lambda_" + constraint
                    ]
            var.update(lambda_diff)

        self.loss_component = Variables(
            ChainMap(
                *[
                    diff[domain_name].weighted_loss(
                        f"{domain_name}_loss",
                        loss_function=self.domain_losses[domain_name],
                    )
                    for domain_name, domain_val in diff.items()
                ]
            )
        )
        self.notify(self, message={Signal.BEFORE_COMPUTE_LOSS: {**self.loss_component}})
        loss = sum(
            {
                domain_name: self.get_sample_domain(domain_name).sigma
                * self.loss_component[f"{domain_name}_loss"]
                for domain_name in diff
            }.values()
        )
        self.notify(
            self,
            message={
                Signal.AFTER_COMPUTE_LOSS: {
                    **self.loss_component,
                    **{"total_loss": loss},
                }
            },
        )
        return loss

    def infer_step(self, domain_attr: Dict[str, List[str]]) -> DomainVariables:
        """Specify a domain and required fields for inference.
        :param domain_attr: A map from a domain name to the list of required outputs on the domain.
        :type domain_attr: Dict[str, List[str]]
        :return: A dict of variables which are required.
        :rtype: Dict[str, Variables]
        """
        samples = self.sample_variables_from_domains()
        in_var, true_out, lambda_out = self.generate_in_out_dict(samples)
        pred_out_sample = self.forward_through_all_graph(in_var, domain_attr)
        return pred_out_sample

    def sample_variables_from_domains(self) -> DomainVariables:
        return {data_node.name: data_node.sample() for data_node in self.sample_domains}

    def save(self):
        """Save parameters of netnodes and the global step to `model.ckpt`."""
        save_path = os.path.join(self.network_dir, "model.ckpt")
        logger.info("save to path: {}".format(os.path.abspath(save_path)))
        save_dict = {
            f"{net_node.name}_dict": net_node.state_dict()
            for net_node in filter(lambda _net: not _net.is_reference, self.netnodes)
        }
        for i, opt in enumerate(self.optimizers):
            save_dict["optimizer_{}_dict".format(i)] = opt.state_dict()
        save_dict["global_step"] = self.global_step
        torch.save(save_dict, save_path)

    def init_load(self):
        for network_dir in self.init_network_dirs:
            save_path = os.path.join(network_dir, "model.ckpt")
            save_dict = torch.load(save_path)
            for net_node in self.netnodes:
                if (
                    f"{net_node.name}_dict" in save_dict.keys()
                    and not net_node.is_reference
                ):
                    net_node.load_state_dict(save_dict[f"{net_node.name}_dict"])
                    logger.info(f"Successfully loading initialization {net_node.name}.")

    def load(self):
        """Load parameters of netnodes and the global step from `model.ckpt`."""
        save_path = os.path.join(self.network_dir, "model.ckpt")
        if not idrlnet.GPU_ENABLED:
            save_dict = torch.load(save_path, map_location=torch.device("cpu"))
        else:
            save_dict = torch.load(save_path)
        # todo: save on CPU, load on GPU
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(save_dict["optimizer_{}_dict".format(i)])
        self.global_step = save_dict["global_step"]
        for net_node in self.netnodes:
            if (
                f"{net_node.name}_dict" in save_dict.keys()
                and not net_node.is_reference
            ):
                net_node.load_state_dict(save_dict[f"{net_node.name}_dict"])
                logger.info(f"Successfully loading {net_node.name}.")

    def configure_optimizers(self):
        """
        Call interfaces of ``Optimizable``
        """
        opt = self.optimizer_config["optimizer"]
        if isinstance(opt, str) and opt in Optimizable.OPTIMIZER_MAP:
            opt = Optimizable.OPTIMIZER_MAP[opt](
                self.trainable_parameters,
                **{k: v for k, v in self.optimizer_config.items() if k != "optimizer"},
            )
        elif isinstance(opt, Callable):
            opt = opt
        else:
            raise NotImplementedError(
                "The optimizer is not implemented. You may use one of the following optimizer:\n"
                + "\n".join(Optimizable.OPTIMIZER_MAP.keys())
                + '\n Example: opt_config=dict(optimizer="Adam", lr=1e-3)'
            )

        lr_scheduler = self.schedule_config["scheduler"]
        if isinstance(lr_scheduler, str) and lr_scheduler in Optimizable.SCHEDULE_MAP:
            lr_scheduler = Optimizable.SCHEDULE_MAP[lr_scheduler](
                opt,
                **{k: v for k, v in self.schedule_config.items() if k != "scheduler"},
            )
        elif isinstance(lr_scheduler, Callable):
            lr_scheduler = lr_scheduler
        else:
            raise NotImplementedError(
                "The scheduler is not implemented. You may use one of the following scheduler:\n"
                + "\n".join(Optimizable.SCHEDULE_MAP.keys())
                + '\n Example: schedule_config=dict(scheduler="ExponentialLR", gamma=0.999'
            )
        self.optimizers = [opt]
        self.schedulers = [lr_scheduler]
