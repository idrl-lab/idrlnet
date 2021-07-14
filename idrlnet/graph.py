"""Define Computational graph"""

import sympy as sp
from typing import List, Dict, Union
from copy import copy
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import math
from idrlnet.variable import Variables
from idrlnet.node import Node
from idrlnet.header import logger, DIFF_SYMBOL
from idrlnet.pde import PdeNode
from idrlnet.net import NetNode

__all__ = ["ComputableNodeList", "Vertex", "VertexTaskPipeline"]
x, y = sp.symbols("x y")
ComputableNodeList = [List[Union[PdeNode, NetNode]]]


class Vertex(Node):
    counter = 0

    def __init__(self, pre=None, next=None, node=None, ntype="c"):
        node = Node() if node is None else node
        self.__dict__ = node.__dict__.copy()
        self.index = type(self).counter
        type(self).counter += 1
        self.pre = pre if pre is not None else set()
        self.next = next if pre is not None else set()
        self.ntype = ntype
        assert self.ntype in ("d", "c", "r")

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return self.index

    def __str__(self):
        info = (
            f"index: {self.index}\n"
            + f"pre: {[node.index for node in self.pre]}\n"
            + f"next: {[node.index for node in self.next]}\n"
        )
        return super().__str__() + info


class VertexTaskPipeline:
    MAX_STACK_ALLOWED = 100000

    @property
    def evaluation_order_list(self):
        return self._evaluation_order_list

    @evaluation_order_list.setter
    def evaluation_order_list(self, evaluation_order_list):
        self._evaluation_order_list = evaluation_order_list

    def __init__(
        self, nodes: ComputableNodeList, invar: Variables, req_names: List[str]
    ):
        self.nodes = nodes
        self.req_names = req_names
        self.computable = set(invar.keys())

        graph_nodes = set(Vertex(node=node) for node in nodes)
        req_name_dict: Dict[str, List[Vertex]] = defaultdict(list)

        self.G = nx.DiGraph()
        self.egde_data = defaultdict(set)
        required_stack = []
        for req_name in req_names:
            final_graph_node = Vertex()
            if DIFF_SYMBOL in req_name:
                final_graph_node.derivatives = (req_name,)
                final_graph_node.inputs = tuple()
            else:
                final_graph_node.inputs = [req_name]
                final_graph_node.derivatives = tuple()
            final_graph_node.outputs = tuple()
            final_graph_node.name = f"<{req_name}>"
            final_graph_node.ntype = "r"
            graph_nodes.add(final_graph_node)
            req_name_dict[req_name].append(final_graph_node)
            required_stack.append(final_graph_node)
            final_graph_node.evaluate = lambda x: x

        logger.info("Constructing computation graph...")
        while len(req_name_dict) > 0:
            to_be_removed = set()
            to_be_added = defaultdict(list)
            if len(required_stack) >= self.MAX_STACK_ALLOWED:
                raise ValueError
            for req_name, current_gn in req_name_dict.items():
                req_name = tuple(req_name.split(DIFF_SYMBOL))
                match_score = -1
                match_gn = None
                for gn in graph_nodes:
                    if gn in current_gn:
                        continue
                    for output in gn.outputs:
                        output = tuple(output.split(DIFF_SYMBOL))
                        if (
                            len(output) <= len(req_name)
                            and req_name[: len(output)] == output
                            and len(output) > match_score
                        ):
                            match_score = len(output)
                            match_gn = gn
                for p_in in invar.keys():
                    p_in = tuple(p_in.split(DIFF_SYMBOL))
                    if (
                        len(p_in) <= len(req_name)
                        and req_name[: len(p_in)] == p_in
                        and len(p_in) > match_score
                    ):
                        match_score = len(p_in)
                        match_gn = None
                        for sub_gn in req_name_dict[DIFF_SYMBOL.join(req_name)]:
                            self.G.add_edge(DIFF_SYMBOL.join(p_in), sub_gn.name)
                if match_score <= 0:
                    raise Exception("Can't be computed: " + DIFF_SYMBOL.join(req_name))
                elif match_gn is not None:
                    for sub_gn in req_name_dict[DIFF_SYMBOL.join(req_name)]:
                        logger.info(
                            f"{sub_gn.name}.{DIFF_SYMBOL.join(req_name)} <---- {match_gn.name}"
                        )
                        match_gn.next.add(sub_gn)
                        self.egde_data[(match_gn.name, sub_gn.name)].add(
                            DIFF_SYMBOL.join(req_name)
                        )
                    required_stack.append(match_gn)
                    for sub_gn in req_name_dict[DIFF_SYMBOL.join(req_name)]:
                        sub_gn.pre.add(match_gn)
                    for p in match_gn.inputs:
                        to_be_added[p].append(match_gn)
                    for p in match_gn.derivatives:
                        to_be_added[p].append(match_gn)
                    for sub_gn in req_name_dict[DIFF_SYMBOL.join(req_name)]:
                        self.G.add_edge(match_gn.name, sub_gn.name)
                to_be_removed.add(DIFF_SYMBOL.join(req_name))
            if len(to_be_removed) == 0 and len(req_name_dict) > 0:
                raise Exception("Can't be computed")
            for p in to_be_removed:
                req_name_dict.pop(p)
                self.computable.add(p)
            for k, v in to_be_added.items():
                if k in req_name_dict:
                    req_name_dict[k].extend(v)
                else:
                    req_name_dict[k] = v
        evaluation_order = []
        while len(required_stack) > 0:
            gn = required_stack.pop()
            if gn not in evaluation_order:
                evaluation_order.append(gn)
                self.computable = self.computable.union(set(gn.outputs))
        self.evaluation_order_list = evaluation_order
        self._graph_node_table = {node.name: node for node in graph_nodes}
        for key in invar:
            node = Vertex()
            node.name = key
            node.outputs = (key,)
            node.inputs = tuple()
            node.ntype = "d"
            self._graph_node_table[key] = node
        logger.info("Computation graph constructed.")

    def operation_order(self, invar: Variables):
        for node in self.evaluation_order_list:
            if not set(node.derivatives).issubset(invar.keys()):
                invar.differentiate_(
                    independent_var=invar, required_derivatives=node.derivatives
                )
            invar.update(
                node.evaluate(
                    {**invar.subset(node.inputs), **invar.subset(node.derivatives)}
                )
            )

    def forward_pipeline(
        self, invar: Variables, req_names: List[str] = None
    ) -> Variables:
        if req_names is None or set(req_names).issubset(set(self.computable)):
            outvar = copy(invar)
            self.operation_order(outvar)
            return outvar.subset(self.req_names if req_names is None else req_names)
        else:
            logger.info("The existing graph fails. Construct a temporary graph...")
            return VertexTaskPipeline(self.nodes, invar, req_names).forward_pipeline(
                invar
            )

    def to_json(self):
        pass

    def display(self, filename: str = None):
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.axis("off")
        pos = nx.spring_layout(self.G, k=10 / (math.sqrt(self.G.order()) + 0.1))
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=list(
                node
                for node in self.G.nodes
                if self._graph_node_table[node].ntype == "c"
            ),
            cmap=plt.get_cmap("jet"),
            node_size=1300,
            node_color="pink",
            alpha=0.5,
        )
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=list(
                node
                for node in self.G.nodes
                if self._graph_node_table[node].ntype == "r"
            ),
            cmap=plt.get_cmap("jet"),
            node_size=1300,
            node_color="green",
            alpha=0.3,
        )
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=list(
                node
                for node in self.G.nodes
                if self._graph_node_table[node].ntype == "d"
            ),
            cmap=plt.get_cmap("jet"),
            node_size=1300,
            node_color="blue",
            alpha=0.3,
        )
        nx.draw_networkx_edges(
            self.G, pos, edge_color="r", arrows=True, arrowsize=30, arrowstyle="-|>"
        )
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            edge_labels={k: ", ".join(v) for k, v in self.egde_data.items()},
            font_size=10,
        )
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()
