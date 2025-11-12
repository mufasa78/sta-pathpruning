import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import List, Optional, Tuple
from .data_structures import Node, TimingPath, EndpointBasedGraph


class Visualizer:
    def __init__(self):
        self.color_scheme = {
            'critical': '#FF4B4B',
            'normal': '#4B9BFF',
            'anchor': '#FFD700',
            'shared': '#90EE90',
            'unique': '#D3D3D3'
        }

    def create_timing_path_graph(self, 
                                 endpoint_graph: EndpointBasedGraph,
                                 paths_to_show: List[TimingPath],
                                 anchor_node: Optional[Node] = None,
                                 max_paths: int = 10) -> go.Figure:
        """Create a cleaner hierarchical timing path visualization"""
        paths_subset = paths_to_show[:max_paths]
        
        # Build node registry with unique positioning
        node_registry = {}
        max_depth = 0
        
        # First pass: identify all unique nodes and their depths
        for path in paths_subset:
            for depth, node in enumerate(path.nodes):
                if node.node_id not in node_registry:
                    node_registry[node.node_id] = {
                        'node': node,
                        'min_depth': depth,
                        'max_depth': depth,
                        'paths': []
                    }
                else:
                    node_registry[node.node_id]['min_depth'] = min(node_registry[node.node_id]['min_depth'], depth)
                    node_registry[node.node_id]['max_depth'] = max(node_registry[node.node_id]['max_depth'], depth)
                
                node_registry[node.node_id]['paths'].append(path.path_id)
                max_depth = max(max_depth, depth)
        
        # Create hierarchical layout
        depth_nodes = {}
        for node_id, info in node_registry.items():
            avg_depth = (info['min_depth'] + info['max_depth']) / 2
            depth_key = int(avg_depth)
            if depth_key not in depth_nodes:
                depth_nodes[depth_key] = []
            depth_nodes[depth_key].append(node_id)
        
        # Position nodes with better spacing
        node_positions = {}
        for depth in sorted(depth_nodes.keys()):
            nodes_at_depth = depth_nodes[depth]
            num_nodes = len(nodes_at_depth)
            for idx, node_id in enumerate(nodes_at_depth):
                y_pos = (idx - num_nodes/2) * 3  # Better vertical spacing
                node_positions[node_id] = (depth * 5, y_pos)  # Better horizontal spacing
        
        # Create separate traces for different node types
        critical_nodes = {'x': [], 'y': [], 'text': [], 'ids': []}
        anchor_nodes = {'x': [], 'y': [], 'text': [], 'ids': []}
        normal_nodes = {'x': [], 'y': [], 'text': [], 'ids': []}
        
        for node_id, info in node_registry.items():
            x, y = node_positions[node_id]
            node = info['node']
            hover_text = (f"Node: {node_id}<br>"
                         f"Slack: {node.slack:.2f} ps<br>"
                         f"Arrival: {node.arrival_time:.2f} ps<br>"
                         f"Paths: {len(info['paths'])}")
            
            if anchor_node and node.node_id == anchor_node.node_id:
                anchor_nodes['x'].append(x)
                anchor_nodes['y'].append(y)
                anchor_nodes['text'].append(hover_text)
                anchor_nodes['ids'].append(node_id)
            elif node.slack < -100:
                critical_nodes['x'].append(x)
                critical_nodes['y'].append(y)
                critical_nodes['text'].append(hover_text)
                critical_nodes['ids'].append(node_id)
            else:
                normal_nodes['x'].append(x)
                normal_nodes['y'].append(y)
                normal_nodes['text'].append(hover_text)
                normal_nodes['ids'].append(node_id)
        
        # Create edges with reduced opacity for cleaner look
        edge_x = []
        edge_y = []
        
        for path in paths_subset:
            for i in range(len(path.nodes) - 1):
                node1_id = path.nodes[i].node_id
                node2_id = path.nodes[i + 1].node_id
                
                if node1_id in node_positions and node2_id in node_positions:
                    x0, y0 = node_positions[node1_id]
                    x1, y1 = node_positions[node2_id]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
        
        # Build figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 100, 100, 0.2)'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add normal nodes
        if normal_nodes['x']:
            fig.add_trace(go.Scatter(
                x=normal_nodes['x'], y=normal_nodes['y'],
                mode='markers',
                marker=dict(size=12, color=self.color_scheme['normal'], 
                           line=dict(width=1, color='white')),
                text=normal_nodes['text'],
                hoverinfo='text',
                name='Normal Nodes',
                showlegend=True
            ))
        
        # Add critical nodes
        if critical_nodes['x']:
            fig.add_trace(go.Scatter(
                x=critical_nodes['x'], y=critical_nodes['y'],
                mode='markers',
                marker=dict(size=14, color=self.color_scheme['critical'],
                           line=dict(width=2, color='white'), symbol='diamond'),
                text=critical_nodes['text'],
                hoverinfo='text',
                name='Critical Nodes',
                showlegend=True
            ))
        
        # Add anchor node
        if anchor_nodes['x']:
            fig.add_trace(go.Scatter(
                x=anchor_nodes['x'], y=anchor_nodes['y'],
                mode='markers',
                marker=dict(size=20, color=self.color_scheme['anchor'],
                           line=dict(width=3, color='orange'), symbol='star'),
                text=anchor_nodes['text'],
                hoverinfo='text',
                name='Anchor Node',
                showlegend=True
            ))
        
        fig.update_layout(
            title=dict(
                text=f'Timing Path Graph - {endpoint_graph.endpoint}<br>' +
                     f'<sub>Showing {len(paths_subset)} paths | {len(node_registry)} unique nodes</sub>',
                font=dict(size=16)
            ),
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
            hovermode='closest',
            plot_bgcolor='#f8f9fa',
            margin=dict(b=40, l=40, r=40, t=80),
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(200, 200, 200, 0.3)',
                zeroline=False, 
                showticklabels=True,
                title='Timing Depth'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                zeroline=False, 
                showticklabels=False
            ),
            height=700
        )

        return fig

    def create_slack_distribution(self, 
                                  endpoint_graph: EndpointBasedGraph,
                                  baseline_paths: Optional[List[TimingPath]] = None,
                                  proposed_paths: Optional[List[TimingPath]] = None) -> go.Figure:
        all_slacks = [p.slack for p in endpoint_graph.paths]

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=all_slacks,
            name='All Paths',
            opacity=0.5,
            nbinsx=50,
            marker_color=self.color_scheme['normal']
        ))

        if baseline_paths:
            baseline_slacks = [p.slack for p in baseline_paths]
            fig.add_trace(go.Scatter(
                x=baseline_slacks,
                y=[0] * len(baseline_slacks),
                mode='markers',
                name='Baseline Top-10',
                marker=dict(size=10, color='blue', symbol='diamond')
            ))

        if proposed_paths:
            proposed_slacks = [p.slack for p in proposed_paths]
            fig.add_trace(go.Scatter(
                x=proposed_slacks,
                y=[0] * len(proposed_slacks),
                mode='markers',
                name='Proposed Top-10',
                marker=dict(size=10, color='red', symbol='star')
            ))

        fig.update_layout(
            title=dict(text='Slack Distribution', font=dict(size=16)),
            xaxis_title='Slack (ps)',
            yaxis_title='Count',
            barmode='overlay',
            height=400
        )

        return fig

    def create_anchor_analysis(self, 
                              endpoint_graph: EndpointBasedGraph,
                              candidates: List[Node],
                              anchor: Node) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Path Coverage by Candidate', 'Slack Distribution by Candidate',
                          'Anchor Selection Score', 'Critical Path Coverage')
        )

        candidate_data = []
        for node in candidates:
            paths_through = endpoint_graph.get_paths_through_node(node)
            critical_paths = endpoint_graph.get_top_k_critical_paths(10)
            critical_coverage = sum(1 for p in paths_through if p in critical_paths)

            candidate_data.append({
                'node_id': node.node_id[:15],
                'coverage': len(paths_through),
                'avg_slack': np.mean([p.slack for p in paths_through]) if paths_through else 0,
                'critical_coverage': critical_coverage,
                'is_anchor': node.node_id == anchor.node_id
            })

        colors = [self.color_scheme['anchor'] if d['is_anchor'] else self.color_scheme['normal'] 
                 for d in candidate_data]

        fig.add_trace(go.Bar(
            x=[d['node_id'] for d in candidate_data],
            y=[d['coverage'] for d in candidate_data],
            marker_color=colors,
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=[d['avg_slack'] for d in candidate_data],
            marker_color=self.color_scheme['normal'],
            showlegend=False
        ), row=1, col=2)

        scores = [d['coverage'] * 0.5 + d['critical_coverage'] * 0.4 - d['avg_slack'] / 1000 * 0.1 
                 for d in candidate_data]
        fig.add_trace(go.Scatter(
            x=[d['node_id'] for d in candidate_data],
            y=scores,
            mode='markers',
            marker=dict(size=10, color=colors),
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=[d['node_id'] for d in candidate_data],
            y=[d['critical_coverage'] for d in candidate_data],
            marker_color=colors,
            showlegend=False
        ), row=2, col=2)

        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=700, title=dict(text='Anchor Node Analysis', font=dict(size=16)))

        return fig

    def create_feature_correlation_heatmap(self, feature_importance: dict) -> go.Figure:
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())

        sorted_indices = np.argsort(importances)[::-1][:15]

        top_features = [features[i] for i in sorted_indices]
        top_importances = [importances[i] for i in sorted_indices]

        fig = go.Figure(data=go.Heatmap(
            z=[top_importances],
            x=top_features,
            colorscale='Viridis',
            showscale=True
        ))

        fig.update_layout(
            title='Top 15 Feature Importance',
            xaxis_title='Features',
            height=300,
            xaxis=dict(tickangle=45)
        )

        return fig