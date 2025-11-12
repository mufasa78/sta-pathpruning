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
        G = nx.DiGraph()

        paths_subset = paths_to_show[:max_paths]

        node_positions = {}
        node_colors = {}
        node_sizes = {}

        for path_idx, path in enumerate(paths_subset):
            for depth, node in enumerate(path.nodes):
                if node.node_id not in G:
                    G.add_node(node.node_id)
                    node_positions[node.node_id] = (depth, -path_idx * 2)

                    if anchor_node and node.node_id == anchor_node.node_id:
                        node_colors[node.node_id] = self.color_scheme['anchor']
                        node_sizes[node.node_id] = 30
                    elif node.slack < -100:
                        node_colors[node.node_id] = self.color_scheme['critical']
                        node_sizes[node.node_id] = 20
                    else:
                        node_colors[node.node_id] = self.color_scheme['normal']
                        node_sizes[node.node_id] = 15

                if depth > 0:
                    prev_node = path.nodes[depth - 1]
                    G.add_edge(prev_node.node_id, node.node_id)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = node_positions[edge[0]]
            x1, y1 = node_positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []

        for node_id in G.nodes():
            x, y = node_positions[node_id]
            node_x.append(x)
            node_y.append(y)
            node_color.append(node_colors[node_id])
            node_size.append(node_sizes[node_id])
            node_text.append(f"{node_id[:20]}...")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=dict(text=f'Timing Path Graph - {endpoint_graph.endpoint}', font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
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