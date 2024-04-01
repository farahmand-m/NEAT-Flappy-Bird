import pygame
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl


def find_layer_index(node, nodes):
    for index, layer in enumerate(nodes):
        if node in layer:
            return index
    raise ValueError('Node not found in prior layers')


def node_coords(node, nodes, x_offset, y_offset, node_radius, layer_spacing, node_spacing):
    layer = find_layer_index(node, nodes)
    num_nodes = len(nodes[layer])
    node_index = nodes[layer].index(node)
    x = (layer - len(nodes) + 1) * layer_spacing + x_offset
    vertical_offset = (num_nodes - 1) * node_spacing
    y = node_index * node_spacing - vertical_offset / 2 + y_offset
    return x, y


def draw_neat_network(screen, network, x, y, node_radius=20, layer_spacing=120, node_spacing=50):
    # Define colors
    hidden_node_color = (198, 198, 198)
    input_node_color = (20, 20, 20)
    output_node_color = (200, 0, 0)

    # Calculate the positions of nodes
    input_nodes = network.input_nodes
    output_nodes = network.output_nodes

    nodes = [input_nodes]
    connections = []

    for node, act_func, agg_func, bias, weight, inputs in network.node_evals:
        max_input_node_layer = -1
        for input_node, conn_weight in inputs:
            connections.append((input_node, node, conn_weight))
            input_node_layer = find_layer_index(input_node, nodes)
            max_input_node_layer = max(max_input_node_layer, input_node_layer)
        node_layer = max_input_node_layer + 1
        if len(nodes) >= node_layer + 1:
            nodes[node_layer].append(node)
        else:
            nodes.append([node])

    for layer in nodes:
        for node in layer:
            coords = node_coords(node, nodes, x, y, node_radius, layer_spacing, node_spacing)
            pygame.draw.circle(screen, (255, 255, 255), coords, node_radius * 1.2)

    if connections:
        scaler = MinMaxScaler().fit([[weight] for start, end, weight in connections])
        for start, end, weight in connections:
            start_coords = node_coords(start, nodes, x, y, node_radius, layer_spacing, node_spacing)
            end_coords = node_coords(end, nodes, x, y, node_radius, layer_spacing, node_spacing)
            weight = scaler.transform([[weight]])
            color = mpl.cm.rainbow(weight)
            color = (color[0, 0, :3] * 255).astype(int).tolist()
            pygame.draw.line(screen, color, start_coords, end_coords, 5)

    for layer in nodes:
        for node in layer:
            coords = node_coords(node, nodes, x, y, node_radius, layer_spacing, node_spacing)
            color = output_node_color if node in output_nodes else input_node_color if node in input_nodes else hidden_node_color
            pygame.draw.circle(screen, color, coords, node_radius)
