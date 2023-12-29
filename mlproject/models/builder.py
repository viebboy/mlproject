"""
builder.py: network builder module
----------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-05-10
* Version: 0.0.1

This is part of the MLProject

License
-------
Apache 2.0 License


"""

from __future__ import annotations
import torch.nn as nn
from collections import OrderedDict
import plotly.graph_objects as go
import colorsys
import networkx as nx
from collections import deque
from loguru import logger
from mlproject.models.nodes import build_node, get_all_node_types


def _assign_layers(adj_matrix):
    num_nodes = len(adj_matrix)
    incoming_edges = [0] * num_nodes
    for j in range(num_nodes):
        for i in range(num_nodes):
            incoming_edges[j] += adj_matrix[i][j]

    # Nodes with no incoming edges are in the first layer
    layers = [-1] * num_nodes
    queue = deque()
    for i in range(num_nodes):
        if incoming_edges[i] == 0:
            queue.append(i)
            layers[i] = 0

    # Assign layers using BFS
    while queue:
        node = queue.popleft()
        for i in range(num_nodes):
            if adj_matrix[node][i]:
                incoming_edges[i] -= 1
                if incoming_edges[i] == 0:
                    queue.append(i)
                    layers[i] = layers[node] + 1

    return layers


def _create_adjacency_matrix(nodes):
    # Dictionary to map node names to their indices
    node_indices = {node["name"]: i for i, node in enumerate(nodes)}

    # Initialize the adjacency matrix with zeros
    size = len(nodes)
    adjacency_matrix = [[0] * size for _ in range(size)]

    # Populate the adjacency matrix
    for i, node in enumerate(nodes):
        inputs = node["input"]

        # If the input is a string, convert it to a list for uniform processing
        if isinstance(inputs, str):
            inputs = [inputs]

        for input_node in inputs:
            if input_node == "previous" and i > 0:
                # Connect to the immediate previous node
                adjacency_matrix[i - 1][i] = 1
            elif input_node in node_indices:
                # Connect to the specified input node
                adjacency_matrix[node_indices[input_node]][i] = 1

    node_layers = _assign_layers(adjacency_matrix)

    for i, node in enumerate(nodes):
        node["layer"] = f"Layer_{node_layers[i]}"

    return adjacency_matrix


def visualize_topology(nodes: list[dict], path: str):
    adj_matrix = _create_adjacency_matrix(nodes)
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for node in nodes:
        G.add_node(node["name"], **node)
    for i, source_node in enumerate(nodes):
        for j, target_node in enumerate(nodes):
            if adj_matrix[i][j] != 0:
                G.add_edge(source_node["name"], target_node["name"])

    # Use Graphviz for layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")

    # Edge traces
    edge_x = []
    edge_y = []
    arrows = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Adding arrow annotation
        arrows.append(
            dict(
                ax=x0,
                ay=y0,
                axref="x",
                ayref="y",
                x=x1,
                y=y1,
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=2,
                arrowwidth=1,
                arrowcolor="#888",
            )
        )

    # Create edge trace with showlegend set to False
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="#888"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    # Determine unique node types and assign a color to each
    node_types = set(node.get("type", "default") for node in nodes)
    type_colors = {"input": "red", "output": "green"}
    hues = [i / len(node_types) for i in range(len(node_types))]
    for i, node_type in enumerate(node_types):
        if node_type not in type_colors:
            # Avoiding hues close to red (0) and green (1/3)
            hue = (hues[i] + 0.1) % 1.0
            if 0.28 < hue < 0.38:
                hue += 0.1
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.7)
            type_colors[node_type] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    # Create node traces based on types
    node_traces = {}
    for node, attr in G.nodes(data=True):
        node_type = attr.get("type", "default")
        color = type_colors[node_type]

        if node_type not in node_traces:
            node_traces[node_type] = {
                "x": [],
                "y": [],
                "hovertext": [],
                "name": node_type,
                "text": [],
                "textcolor": [],
            }
        node_traces[node_type]["x"].append(pos[node][0])
        node_traces[node_type]["y"].append(pos[node][1])
        node_traces[node_type]["hovertext"].append(
            "<br>".join(
                [f"{key}: {value}" for key, value in attr.items() if key != "layer"]
            )
        )

        # Custom text and color for input and output nodes
        node_label = node
        text_color = color
        if attr.get("input") == "input":
            node_label += " (IN)"
            text_color = "red"
        elif attr.get("is_output"):
            node_label += " (OUT)"
            text_color = "green"

        node_traces[node_type]["text"].append(node_label)
        node_traces[node_type]["textcolor"].append(text_color)

    # Create figure and add node traces
    fig = go.Figure(data=[edge_trace])
    for node_type, trace_data in node_traces.items():
        node_trace = go.Scatter(
            x=trace_data["x"],
            y=trace_data["y"],
            mode="markers+text",
            text=trace_data["text"],
            textposition="bottom center",
            hoverinfo="text",
            marker=dict(
                showscale=False, color=type_colors[node_type], size=10, line_width=2
            ),
            hovertext=trace_data["hovertext"],
            textfont=dict(color=trace_data["textcolor"]),
            name=node_type,  # This is used for the legend entry
        )
        fig.add_trace(node_trace)

    # Layout settings
    fig.update_layout(
        title="<br>Network Topology",
        titlefont_size=16,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=arrows,
    )

    if path is not None:
        assert path.endswith(".html")
        fig.write_html(path)
    else:
        fig.show()


class Builder(nn.Module):
    def __init__(self, nodes: list):
        super().__init__()
        self._verify_config(nodes)
        self._topology = nodes
        self._build_nodes(nodes)
        self._node_metadata, self._output_names = self._build_node_metadata(nodes)

    def _verify_node_keys(self, node: dict):
        required_keys = ["name", "type", "input", "is_output"]
        for key in required_keys:
            if key not in node:
                logger.error(f"Node is missing required key: {key} | node: {node}")
                raise ValueError(f"Node is missing required key: {key} | node: {node}")

        # verify if node name contains dot
        if "." in node["name"]:
            logger.error(
                f'Node name cannot contain "." | node name: {node["name"]} | node: {node}'
            )
            raise ValueError(
                f'Node name cannot contain "." | node name: {node["name"]} | node: {node}'
            )

    def _verify_config(self, nodes: list):
        # get supported nodes
        supported_nodes = get_all_node_types()

        # check if node types are supported
        nb_output = 0
        names = []
        prev_node_name = "input"
        for node in nodes:
            # check if node has required
            self._verify_node_keys(node)

            if node["type"] not in supported_nodes:
                logger.error("Unsupported node type: {}".format(node["type"]))
                logger.warning(f"Supported node types: {supported_nodes}")
                logger.info(
                    'consider registering your node using "mlproject.models.nodes.register_node()"'
                )
                raise ValueError("Unsupported node type: {}".format(node["type"]))

            nb_output += int(node["is_output"])

            if node["name"] in names:
                logger.error("Duplicate node name: {}".format(node["name"]))
                raise ValueError("Duplicate node name: {}".format(node["name"]))

            if node["input"] == "previous":
                node["input"] = prev_node_name
            prev_node_name = node["name"]

        # check if there is at least one output node
        if nb_output == 0:
            raise ValueError("None of the provided nodes is an output node")

        # check if the input of each node exists in the previous nodes
        for idx, node in enumerate(nodes):
            if isinstance(node["input"], str) and node["input"] != "input":
                prev_node_names = [nodes[k]["name"] for k in range(idx)]
                if node["input"] not in prev_node_names:
                    logger.error(
                        f"Node at index {idx} has an input that does not exist in the previous nodes"
                    )
                    logger.error(f'node name: {node["name"]} | input: {node["input"]}')
                    logger.error(f"previous node names: {prev_node_names}")
                    raise ValueError(
                        f"Node at index {idx} has an input that does not exist in the previous nodes"
                    )
            elif isinstance(node["input"], (tuple, list)):
                prev_node_names = [nodes[k]["name"] for k in range(idx)]
                for input_node in node["input"]:
                    if input_node not in prev_node_names:
                        logger.error(
                            f"Node at index {idx} has an input that does not exist in the previous nodes"
                        )
                        logger.error(
                            f'node name: {node["name"]} | input: {node["input"]}'
                        )
                        logger.error(f"previous node names: {prev_node_names}")
                        raise ValueError(
                            f"Node at index {idx} has an input that does not exist in the previous nodes"
                        )

    def _build_nodes(self, nodes):
        all_nodes = OrderedDict()
        for node_config in nodes:
            try:
                node = build_node(node_config)
            except Exception as error:
                logger.error(f'failed to build node with name: {node_config["name"]}')
                logger.error(f"node config: {node_config}")
                logger.error(f"faced the following error: {str(error)}")
                raise error

            all_nodes[node_config["name"]] = node

        self.nodes = nn.ModuleDict(all_nodes)

    def _build_node_metadata(self, nodes):
        metadata = []
        output_names = []
        # loop through all nodes and keep track of names
        # and whether the output of this node needs to be retained
        for node in nodes:
            metadata.append(
                {
                    "name": node["name"],
                    "input": node["input"],
                    "retain": node["is_output"],
                }
            )
            if node["is_output"]:
                output_names.append(node["name"])

        # for node that has inputs containing element from
        # non-immediate-previous nodes
        # we need to retain the output of the input node
        prev_node = "input"
        retain = []
        for node in metadata:
            if node["input"] == "previous":
                node["input"] = prev_node
            else:
                if node["input"] != prev_node:
                    if isinstance(node["input"], str):
                        retain.append(node["input"])
                    else:
                        retain.extend(node["input"])

        for node in metadata:
            if node["name"] in retain:
                node["retain"] = True

        return metadata, output_names

    def forward(self, inputs):
        data = {"input": inputs}
        prev_node = "input"
        prev_output = inputs

        for node, metadata in zip(self.nodes.values(), self._node_metadata):
            if metadata["input"] == prev_node:
                # if the previous node is the input of the current node
                current_output = node(prev_output)
            else:
                # multiple inputs or the previous node is not the input of the current node
                if isinstance(metadata["input"], str):
                    current_output = node(data[metadata["input"]])
                elif isinstance(metadata["input"], (tuple, list)):
                    inputs = [data[name] for name in metadata["input"]]
                    current_output = node(*inputs)
                else:
                    raise ValueError(
                        '"input" field of a node must be a string or a tuple/list of strings'
                    )

            prev_output = current_output
            prev_node = metadata["name"]
            if metadata["retain"]:
                data[metadata["name"]] = current_output

        return [data[name] for name in self._output_names]

    def _get_node(self, name: str):
        "return node by name"
        if name not in self.nodes:
            raise ValueError(f"Node with name {name} does not exist")
        return self.nodes[name]

    def topology(self):
        return self._topology

    def visualize(self, path: str = None):
        visualize_topology(self._topology, path)
