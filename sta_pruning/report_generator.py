import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from datetime import datetime


class ReportGenerator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_summary_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        summary = pd.DataFrame({
            'Metric': [
                'Total Endpoints',
                'Average MSE',
                'Average MAE',
                'Average Speedup',
                'Average Path Overlap',
                'Min Speedup',
                'Max Speedup',
                'Std Speedup',
                'Total Processing Time (s)',
                'Avg Time per Endpoint (ms)'
            ],
            'Value': [
                len(results_df),
                f"{results_df['mse'].mean():.2e}",
                f"{results_df['mae'].mean():.2e}",
                f"{results_df['speedup'].mean():.2f}×",
                f"{results_df['path_overlap'].mean()*100:.1f}%",
                f"{results_df['speedup'].min():.2f}×",
                f"{results_df['speedup'].max():.2f}×",
                f"{results_df['speedup'].std():.2f}",
                f"{results_df['proposed_time'].sum():.2f}",
                f"{results_df['proposed_time'].mean()*1000:.2f}"
            ]
        })
        return summary
    
    def generate_design_comparison_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        design_stats = results_df.groupby('design').agg({
            'endpoint': 'count',
            'num_paths': 'mean',
            'mse': 'mean',
            'mae': 'mean',
            'speedup': 'mean',
            'path_overlap': 'mean',
            'baseline_time': 'sum',
            'proposed_time': 'sum'
        }).reset_index()
        
        design_stats.columns = [
            'Design',
            'Endpoints',
            'Avg Paths',
            'MSE',
            'MAE',
            'Speedup',
            'Path Overlap',
            'Total Baseline Time (s)',
            'Total Proposed Time (s)'
        ]
        
        design_stats['MSE'] = design_stats['MSE'].apply(lambda x: f"{x:.2e}")
        design_stats['MAE'] = design_stats['MAE'].apply(lambda x: f"{x:.2e}")
        design_stats['Speedup'] = design_stats['Speedup'].apply(lambda x: f"{x:.2f}×")
        design_stats['Path Overlap'] = design_stats['Path Overlap'].apply(lambda x: f"{x*100:.1f}%")
        design_stats['Avg Paths'] = design_stats['Avg Paths'].apply(lambda x: f"{x:.0f}")
        design_stats['Total Baseline Time (s)'] = design_stats['Total Baseline Time (s)'].apply(lambda x: f"{x:.2f}")
        design_stats['Total Proposed Time (s)'] = design_stats['Total Proposed Time (s)'].apply(lambda x: f"{x:.2f}")
        
        return design_stats
    
    def create_comprehensive_plots(self, results_df: pd.DataFrame) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'MAE Distribution',
                'Speedup Distribution',
                'Path Overlap Distribution',
                'MSE vs MAE',
                'Speedup vs Number of Paths',
                'Processing Time Breakdown'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'box'}]
            ]
        )
        
        fig.add_trace(go.Histogram(
            x=results_df['mae'],
            name='MAE',
            nbinsx=30,
            marker_color='blue'
        ), row=1, col=1)
        
        fig.add_trace(go.Histogram(
            x=results_df['speedup'],
            name='Speedup',
            nbinsx=30,
            marker_color='green'
        ), row=1, col=2)
        
        fig.add_trace(go.Histogram(
            x=results_df['path_overlap'],
            name='Path Overlap',
            nbinsx=30,
            marker_color='orange'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=results_df['mse'],
            y=results_df['mae'],
            mode='markers',
            name='MSE vs MAE',
            marker=dict(color='red', size=5)
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=results_df['num_paths'],
            y=results_df['speedup'],
            mode='markers',
            name='Speedup vs Paths',
            marker=dict(color='purple', size=5)
        ), row=3, col=1)
        
        fig.add_trace(go.Box(
            y=results_df['proposed_time'],
            name='Proposed Time',
            marker_color='blue'
        ), row=3, col=2)
        
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="Comprehensive Performance Analysis"
        )
        
        return fig
    
    def create_accuracy_vs_performance_plot(self, results_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results_df['speedup'],
            y=results_df['mae'],
            mode='markers',
            marker=dict(
                size=10,
                color=results_df['path_overlap'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Path Overlap")
            ),
            text=[f"Design: {d}<br>Endpoints: {e}" 
                  for d, e in zip(results_df['design'], results_df['endpoint'])],
            hovertemplate='<b>%{text}</b><br>Speedup: %{x:.2f}×<br>MAE: %{y:.2e}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Accuracy vs Performance Trade-off',
            xaxis_title='Speedup Factor',
            yaxis_title='Mean Absolute Error (MAE)',
            height=500
        )
        
        return fig
    
    def create_statistical_summary_plot(self, results_df: pd.DataFrame) -> go.Figure:
        metrics = ['mse', 'mae', 'speedup', 'path_overlap']
        metric_labels = ['MSE', 'MAE', 'Speedup', 'Path Overlap']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_labels
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, label, pos in zip(metrics, metric_labels, positions):
            fig.add_trace(go.Box(
                y=results_df[metric],
                name=label,
                boxmean='sd'
            ), row=pos[0], col=pos[1])
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Statistical Distribution of Key Metrics"
        )
        
        return fig
    
    def export_to_csv(self, results_df: pd.DataFrame, filename: str = None) -> str:
        if filename is None:
            filename = f"sta_benchmark_results_{self.timestamp}.csv"
        
        results_df.to_csv(filename, index=False)
        return filename
    
    def generate_markdown_report(self, 
                                 results_df: pd.DataFrame,
                                 summary_stats: Dict[str, Any]) -> str:
        report = f"""# STA Path Pruning - Benchmark Report
        
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

- **Total Endpoints Analyzed:** {summary_stats.get('total_endpoints', 0)}
- **Average MAE:** {summary_stats.get('avg_mae', 0):.2e} (Target: <1.6e-04)
- **Average Speedup:** {summary_stats.get('avg_speedup', 0):.2f}× (Target: ≥1.3×)
- **Average Path Overlap:** {summary_stats.get('avg_path_overlap', 0)*100:.1f}% (Target: >90%)

## Performance Metrics

### Accuracy
- Mean Squared Error (MSE): {summary_stats.get('avg_mse', 0):.2e}
- Mean Absolute Error (MAE): {summary_stats.get('avg_mae', 0):.2e}

### Speed
- Minimum Speedup: {summary_stats.get('min_speedup', 0):.2f}×
- Maximum Speedup: {summary_stats.get('max_speedup', 0):.2f}×
- Standard Deviation: {summary_stats.get('std_speedup', 0):.2f}

### Coverage
- Average Path Overlap: {summary_stats.get('avg_path_overlap', 0)*100:.1f}%

## Design-Level Results

"""
        
        design_stats = results_df.groupby('design').agg({
            'mse': 'mean',
            'mae': 'mean',
            'speedup': 'mean',
            'path_overlap': 'mean'
        })
        
        for design in design_stats.index:
            stats = design_stats.loc[design]
            report += f"### {design}\n"
            report += f"- MSE: {stats['mse']:.2e}\n"
            report += f"- MAE: {stats['mae']:.2e}\n"
            report += f"- Speedup: {stats['speedup']:.2f}×\n"
            report += f"- Path Overlap: {stats['path_overlap']*100:.1f}%\n\n"
        
        report += """## Conclusions

"""
        
        if summary_stats.get('avg_mae', float('inf')) < 1.6e-04:
            report += "✅ **Accuracy Target Met:** MAE is below the 1.6e-04 threshold.\n"
        else:
            report += "⚠️ **Accuracy Target Missed:** MAE exceeds the 1.6e-04 threshold.\n"
        
        if summary_stats.get('avg_speedup', 0) >= 1.3:
            report += "✅ **Speedup Target Met:** Average speedup exceeds 1.3×.\n"
        else:
            report += "⚠️ **Speedup Target Missed:** Average speedup is below 1.3×.\n"
        
        if summary_stats.get('avg_path_overlap', 0) >= 0.9:
            report += "✅ **Coverage Target Met:** Path overlap exceeds 90%.\n"
        else:
            report += "⚠️ **Coverage Target Missed:** Path overlap is below 90%.\n"
        
        return report
    
    def save_markdown_report(self, report: str, filename: str = None) -> str:
        if filename is None:
            filename = f"sta_benchmark_report_{self.timestamp}.md"
        
        with open(filename, 'w') as f:
            f.write(report)
        
        return filename
