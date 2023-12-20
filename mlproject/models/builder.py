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

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from loguru import logger
from mlproject.models.nodes import get_supported_node_names, build_node


class Builder(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.verify_config(nodes)

        self.build_nodes(nodes)
        self.node_metadata, self.output_names = self.build_node_metadata(nodes)

    def verify_config(self, nodes):
        supported_nodes = get_supported_node_names()

        # check if node types are supported
        nb_output = 0
        names = []
        prev_node_name = 'input'
        for node in nodes:
            if node['type'] not in supported_nodes:
                logger.error('Unsupported node type: {}'.format(node['type']))
                logger.warning(f'Supported node types: {supported_nodes}')
                raise ValueError('Unsupported node type: {}'.format(node['type']))
            nb_output += int(node['is_output'])
            if node['name'] in names:
                logger.error('Duplicate node name: {}'.format(node['name']))
                raise ValueError('Duplicate node name: {}'.format(node['name']))

            if node['input'] == 'previous':
                node['input'] = prev_node_name
            prev_node_name = node['name']

        # check if there is at least one output node
        if nb_output == 0:
            raise ValueError('None of the provided nodes is an output node')

        # check if the input of each node exists in the previous nodes
        for idx, node in enumerate(nodes):
            if isinstance(node['input'], str) and node['input'] != 'input':
                prev_node_names = [nodes[k]['name'] for k in range(idx)]
                if node['input'] not in prev_node_names:
                    logger.error(f'Node at index {idx} has an input that does not exist in the previous nodes')
                    logger.error(f'node name: {node["name"]} | input: {node["input"]}')
                    logger.error(f'previous node names: {prev_node_names}')
                    raise ValueError(f'Node at index {idx} has an input that does not exist in the previous nodes')
            elif isinstance(node['input'], (tuple, list)):
                prev_node_names = [nodes[k]['name'] for k in range(idx)]
                for input_node in node['input']:
                    if input_node not in prev_node_names:
                        logger.error(f'Node at index {idx} has an input that does not exist in the previous nodes')
                        logger.error(f'node name: {node["name"]} | input: {node["input"]}')
                        logger.error(f'previous node names: {prev_node_names}')
                        raise ValueError(f'Node at index {idx} has an input that does not exist in the previous nodes')

    def build_nodes(self, nodes):
        all_nodes = OrderedDict()
        for node_config in nodes:
            try:
                node = build_node(node_config)
            except Exception as error:
                logger.error(f'failed to build node with name: {node_config["name"]}')
                logger.error(f'node config: {node_config}')
                logger.error(f'faced the following error: {str(error)}')
                raise error

            all_nodes[node_config['name']] = node

        self.nodes = nn.ModuleDict(all_nodes)


    def build_node_metadata(self, nodes):
        metadata = []
        output_names = []
        # loop through all nodes and keep track of names
        # and whether the output of this node needs to be retained
        for node in nodes:
            metadata.append({
                'name': node['name'],
                'input': node['input'],
                'retain': node['is_output'],
            })
            if node['is_output']:
                output_names.append(node['name'])


        # for node that has inputs containing element from
        # non-immediate-previous nodes
        # we need to retain the output of the input node
        prev_node = 'input'
        retain = []
        for node in metadata:
            if node['input'] == 'previous':
                node['input'] = prev_node
            else:
                if node['input'] != prev_node:
                    if isinstance(node['input'], str):
                        retain.append(node['input'])
                    else:
                        retain.extend(node['input'])

        for node in metadata:
            if node['name'] in retain:
                node['retain'] = True

        return metadata, output_names

    def forward(self, inputs):
        data = {'input': inputs}
        prev_node = 'input'
        prev_output = inputs

        for node, metadata in zip(self.nodes, self.node_metadata):
            if metadata['input'] == prev_node:
                # if the previous node is the input of the current node
                current_output = node(prev_output)
            else:
                # multiple inputs or the previous node is not the input of the current node
                if isinstance(metadata['input'], str):
                    current_output = node(data[metadata['input']])
                elif isinstance(metadata['input'], (tuple, list)):
                    inputs = [data[name] for name in metadata['input']]
                    current_output = node(*inputs)
                else:
                    raise ValueError('"input" field of a node must be a string or a tuple/list of strings')

            prev_output = current_output
            prev_node = metadata['name']
            if metadata['retain']:
                data[metadata['name']] = current_output

        return [data[name] for name in self.output_names]
