"""
Neural Network Visualizer
=========================
Real-time visualization of a NEAT genome's neural network with dynamic 
topological layering, centered layout, and clean typography.
"""

import pygame
import math
import neat

# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

class NeuralNetworkVisualizer:
    """Visualizes a NEAT genome's neural network in real-time with dynamic layering."""

    # UI Theme Colors
    BG_COLOR = (25, 25, 35)
    PANEL_BORDER = (60, 60, 80)
    HEADER_LINE = (45, 45, 60)
    LEGEND_BG = (18, 18, 26)
    LEGEND_BORDER = (50, 50, 70)
    
    NODE_COLOR = (200, 200, 200)
    NODE_BORDER = (110, 110, 130)
    POS_WEIGHT_COLOR = (50, 150, 255)   # Blue = Excitatory
    NEG_WEIGHT_COLOR = (255, 80, 80)    # Red = Inhibitory
    TEXT_COLOR = (235, 235, 240)
    LABEL_COLOR = (170, 170, 190)
    ACTIVE_GLOW = (255, 220, 100)       # Yellow glow on active nodes

    def __init__(self, genome, config, x=0, y=0, width=400, height=800):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.font_tiny = pygame.font.SysFont("comicsans", 12)
        self.font_small = pygame.font.SysFont("comicsans", 15)
        self.font_title = pygame.font.SysFont("comicsans", 22, bold=True)

        self.genome = genome
        self.config = config
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)

        self.nodes = {}        # node_key -> {x, y, layer, activation}
        self.connections = []  # list of {from, to, weight}
        self.node_layers = {}  # layer_idx -> list of node_keys

        self._build_network()

    def _build_network(self):
        """Extract nodes and connections, computing topological layer depths."""
        input_keys = list(self.config.genome_config.input_keys)
        output_keys = list(self.config.genome_config.output_keys)
        hidden_keys = [k for k in self.genome.nodes if k not in output_keys]

        self.nodes = {}
        self.connections = []

        enabled_conns = {
            conn_key: conn for conn_key, conn in self.genome.connections.items()
            if conn.enabled
        }

        # Recursive depth calculation for hidden nodes
        memo = {k: 0 for k in input_keys}

        def get_layer(node_key, visited=None):
            if visited is None:
                visited = set()
            if node_key in memo:
                return memo[node_key]
            if node_key in visited:
                return 0
            visited.add(node_key)

            parents = [in_k for (in_k, out_k) in enabled_conns.keys() if out_k == node_key]
            layer = 1 if not parents else 1 + max(get_layer(p, visited.copy()) for p in parents)
            memo[node_key] = layer
            return layer

        hidden_depths = {hk: get_layer(hk) for hk in hidden_keys}
        max_hidden_layer = max(hidden_depths.values()) if hidden_depths else 0
        output_layer = max_hidden_layer + 1

        num_layers = output_layer + 1
        self.node_layers = {i: [] for i in range(num_layers)}

        for nk in input_keys:
            self.node_layers[0].append(nk)
            self.nodes[nk] = {'layer': 0, 'activation': 0.0, 'bias': 0.0}

        for hk in hidden_keys:
            layer_idx = hidden_depths[hk]
            self.node_layers[layer_idx].append(hk)
            self.nodes[hk] = {'layer': layer_idx, 'activation': 0.0, 'bias': self.genome.nodes[hk].bias}

        for ok in output_keys:
            self.node_layers[output_layer].append(ok)
            self.nodes[ok] = {'layer': output_layer, 'activation': 0.0, 'bias': self.genome.nodes[ok].bias}

        for (in_key, out_key), conn in enabled_conns.items():
            if in_key in self.nodes and out_key in self.nodes:
                self.connections.append({
                    'from': in_key,
                    'to': out_key,
                    'weight': conn.weight
                })

        self._compute_positions()

    def _compute_positions(self):
        """Compute (x, y) coordinates for nodes with explicit header/footer clearance."""
        top_margin = 85      # Clearance for title header
        bottom_margin = 75   # Clearance for footer legend
        left_margin = 85     # Room for left input text
        right_margin = 75    # Room for right output text

        usable_w = self.width - (left_margin + right_margin)
        usable_h = self.height - (top_margin + bottom_margin)

        num_layers = len(self.node_layers)
        max_node_spacing = 110.0  # Cap vertical distance between nodes

        for layer_idx, node_keys in self.node_layers.items():
            n = len(node_keys)
            if n == 0:
                continue

            # X Coordinate per layer column
            if num_layers > 1:
                x = self.x + left_margin + (layer_idx / (num_layers - 1)) * usable_w
            else:
                x = self.x + left_margin + usable_w / 2.0

            # Y Coordinates - vertically centered in usable space
            layer_height = min((n - 1) * max_node_spacing, usable_h)
            start_y = self.y + top_margin + (usable_h - layer_height) / 2.0

            for i, nk in enumerate(node_keys):
                if n == 1:
                    y = self.y + top_margin + usable_h / 2.0
                else:
                    y = start_y + (i / (n - 1)) * layer_height

                self.nodes[nk]['x'] = x
                self.nodes[nk]['y'] = y

    def update(self, inputs):
        """Update node activations from network input feed."""
        output = self.network.activate(inputs)

        input_keys = list(self.config.genome_config.input_keys)
        output_keys = list(self.config.genome_config.output_keys)

        for i, key in enumerate(input_keys):
            if i < len(inputs):
                self.nodes[key]['activation'] = float(inputs[i])

        for i, key in enumerate(output_keys):
            if i < len(output):
                self.nodes[key]['activation'] = float(output[i])

        hidden_keys = [k for k in self.nodes if k not in input_keys and k not in output_keys]
        for hk in hidden_keys:
            total = self.nodes[hk].get('bias', 0.0)
            for conn in self.connections:
                if conn['to'] == hk and conn['from'] in self.nodes:
                    total += self.nodes[conn['from']]['activation'] * conn['weight']
            self.nodes[hk]['activation'] = math.tanh(total)

    def draw(self, surface):
        """Draw the neural network visualizer onto the surface."""
        # Panel Background & Border
        panel_rect = pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))
        pygame.draw.rect(surface, self.BG_COLOR, panel_rect)
        pygame.draw.rect(surface, self.PANEL_BORDER, panel_rect, 2)

        # Title & Divider Line
        title = self.font_title.render("Neural Network", True, self.TEXT_COLOR)
        surface.blit(title, (int(self.x + 18), int(self.y + 16)))
        pygame.draw.line(
            surface, self.HEADER_LINE,
            (int(self.x + 15), int(self.y + 52)),
            (int(self.x + self.width - 15), int(self.y + 52)),
            1
        )

        # Draw Connection Lines
        for conn in self.connections:
            from_node = self.nodes.get(conn['from'])
            to_node = self.nodes.get(conn['to'])
            if from_node is None or to_node is None:
                continue

            weight = conn['weight']
            color = self.POS_WEIGHT_COLOR if weight >= 0 else self.NEG_WEIGHT_COLOR
            intensity = min(1.0, abs(weight) / 3.0)

            r = int(color[0] * intensity + 30 * (1 - intensity))
            g = int(color[1] * intensity + 30 * (1 - intensity))
            b = int(color[2] * intensity + 30 * (1 - intensity))

            thickness = min(5, max(1, int(abs(weight) * 1.8)))

            pygame.draw.line(
                surface, (r, g, b),
                (int(from_node['x']), int(from_node['y'])),
                (int(to_node['x']), int(to_node['y'])),
                thickness
            )

        # Draw Node Circles
        node_radius = 14

        for key, node in self.nodes.items():
            nx, ny = node['x'], node['y']
            activation = node['activation']

            act_norm = max(0.0, min(1.0, (activation + 1.0) / 2.0))
            base_r = int(50 + act_norm * 150)
            base_g = int(50 + act_norm * 170)
            base_b = int(70 + act_norm * 130)

            # Active Glow Ring
            if abs(activation) > 0.3:
                glow_r = node_radius + 6
                glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
                glow_alpha = min(255, int(80 * min(abs(activation), 1.0)))
                pygame.draw.circle(glow_surf, (*self.ACTIVE_GLOW, glow_alpha), (glow_r, glow_r), glow_r)
                surface.blit(glow_surf, (int(nx - glow_r), int(ny - glow_r)))

            pygame.draw.circle(surface, (base_r, base_g, base_b), (int(nx), int(ny)), node_radius)
            pygame.draw.circle(surface, self.NODE_BORDER, (int(nx), int(ny)), node_radius, 2)

            # Draw bias label below non-input nodes
            bias = node.get('bias', 0.0)
            if abs(bias) > 0.001:
                bias_text = self.font_tiny.render(f"b={bias:+.2f}", True, self.LABEL_COLOR)
                bias_rect = bias_text.get_rect(midtop=(int(nx), int(ny) + node_radius + 3))
                surface.blit(bias_text, bias_rect)

        self._draw_labels(surface)

    def _draw_labels(self, surface):
        """Draw side-aligned input/output node labels and footer legend."""
        #  input_labels = ["Bird Y", "Dist Top", "Dist Bot"]
        input_labels = ["Bird Y", "Dist Top", "Dist Bot", "Dist Pipe"]
        output_labels = ["Jump"]

        # Input Node Labels (Right-aligned to the left of input nodes)
        input_keys = list(self.config.genome_config.input_keys)
        for i, key in enumerate(input_keys):
            node = self.nodes[key]
            label = input_labels[i] if i < len(input_labels) else f"In {i}"
            text = self.font_small.render(label, True, self.LABEL_COLOR)
            rect = text.get_rect(midright=(int(node['x'] - 20), int(node['y'])))
            surface.blit(text, rect)

        # Output Node Labels (Left-aligned to the right of output nodes)
        output_keys = list(self.config.genome_config.output_keys)
        for i, key in enumerate(output_keys):
            node = self.nodes[key]
            label = output_labels[i] if i < len(output_labels) else f"Out {i}"
            text = self.font_small.render(label, True, self.LABEL_COLOR)
            rect = text.get_rect(midleft=(int(node['x'] + 20), int(node['y'])))
            surface.blit(text, rect)

        # Footer Legend Box
        legend_rect = pygame.Rect(
            int(self.x + 15),
            int(self.y + self.height - 58),
            int(self.width - 30),
            44
        )
        pygame.draw.rect(surface, self.LEGEND_BG, legend_rect, border_radius=6)
        pygame.draw.rect(surface, self.LEGEND_BORDER, legend_rect, 1, border_radius=6)

        txt1 = self.font_tiny.render("Blue = +Weight   Red = -Weight", True, self.LABEL_COLOR)
        txt2 = self.font_tiny.render("Glow = Active Node", True, self.LABEL_COLOR)
        surface.blit(txt1, (legend_rect.x + 12, legend_rect.y + 7))
        surface.blit(txt2, (legend_rect.x + 12, legend_rect.y + 23))

    def set_genome(self, genome):
        """Re-initialize structure when leader genome changes."""
        self.genome = genome
        self.network = neat.nn.FeedForwardNetwork.create(genome, self.config)
        self.nodes = {}
        self.connections = []
        self.node_layers = {}
        self._build_network()