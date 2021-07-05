"""shortcut for API"""
from idrlnet.geo_utils import *
from idrlnet.architecture import *
from idrlnet.pde_op import *
from idrlnet.net import NetNode
from idrlnet.data import get_data_node, DataNode, get_data_nodes, datanode, SampleDomain
from idrlnet.pde import ExpressionNode
from idrlnet.solver import Solver
from idrlnet.callbacks import GradientReceiver
from idrlnet.receivers import Receiver, Signal
from idrlnet.variable import Variables, export_var
from idrlnet.header import logger
from idrlnet import GPU_ENABLED
