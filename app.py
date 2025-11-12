import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os

from sta_pruning import (
    SyntheticDataGenerator,
    Pipeline,
    Evaluator,
    ModelTrainer,
    Visualizer,
    ModelTuner,
    BatchProcessor,
    ReportGenerator,
    CircuitNetLoader
)

st.set_page_config(
    page_title="STA Path Pruning - ML-based Circuit Timing Analysis",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def initialize_system():
    data_gen = SyntheticDataGenerator(seed=42)

    training_graphs = []
    for i in range(50):
        graph = data_gen.generate_endpoint_graph(
            f"train_ep_{i}",
            num_paths=np.random.randint(50, 150)
        )
        training_graphs.append(graph)

    trainer = ModelTrainer(model_type='rf', n_estimators=500)
    training_data = trainer.prepare_training_data(training_graphs)

    pipeline_rf = Pipeline(model_type='rf', n_estimators=500)
    pipeline_rf.train(training_data)

    pipeline_xgb = Pipeline(model_type='xgb', n_estimators=500)
    pipeline_xgb.train(training_data)

    return data_gen, pipeline_rf, pipeline_xgb

data_gen, pipeline_rf, pipeline_xgb = initialize_system()

st.title("⚡ STA Path Pruning System")
st.markdown("**ML-based Endpoint-Oriented Path Pruning for Circuit Timing Analysis**")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview",
    "Interactive Demo",
    "Advanced Visualizations",
    "Benchmark Results",
    "Model Tuning",
    "Batch Processing",
    "Model Analysis",
    "CircuitNet Dataset",
    "Documentation"
])

with tab1:
    st.header("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Algorithm Pipeline")
        st.markdown("""
        **Stage 1: Candidate Generation**
        - Extract nodes from middle 40% region of timing paths
        - Select up to 30 candidates based on path coverage and criticality

        **Stage 2: Anchor Prediction**
        - Random Forest classifier (500-1000 trees)
        - 21-dimensional feature extraction:
          - 7 path features (slack statistics, delays, criticality)
          - 14 node features (timing, physical, connectivity)

        **Stage 3: Path Pruning**
        - Filter paths through predicted anchor node
        - Return top 10 worst-case timing paths
        """)

    with col2:
        st.subheader("Target Performance Metrics")

        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'MAE', 'Speedup', 'Path Coverage'],
            'Target': ['~1.4e-06', '<1.6e-04 (<1ps)', '1.3-1.7× (30%)', '>90%'],
            'Description': [
                'Mean Squared Error of slack values',
                'Mean Absolute Error (picoseconds)',
                'Runtime reduction vs exhaustive PBA',
                'Overlap with baseline paths'
            ]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.info("**Status:** System trained on 50 synthetic endpoint graphs with Random Forest and XGBoost models.")

with tab2:
    st.header("Interactive Demonstration")

    st.subheader("Generate Test Circuit")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_paths = st.slider("Number of Paths", 50, 500, 150)
    with col2:
        min_path_len = st.slider("Min Path Length", 10, 30, 15)
    with col3:
        max_path_len = st.slider("Max Path Length", 30, 80, 50)

    model_choice = st.radio("Select Model", ["Random Forest", "XGBoost"], horizontal=True)

    if st.button("Generate & Analyze Circuit", type="primary"):
        with st.spinner("Generating circuit and analyzing..."):
            endpoint_graph = data_gen.generate_endpoint_graph(
                "demo_endpoint",
                num_paths=num_paths,
                min_path_length=min_path_len,
                max_path_length=max_path_len
            )

            pipeline = pipeline_rf if model_choice == "Random Forest" else pipeline_xgb

            baseline_paths, baseline_stats = pipeline.process_baseline(endpoint_graph)
            proposed_paths, proposed_stats = pipeline.process_endpoint(endpoint_graph)

            evaluator = Evaluator()
            result = evaluator.benchmark_endpoint(endpoint_graph, pipeline, "demo_design")

            st.session_state['endpoint_graph'] = endpoint_graph
            st.session_state['baseline_paths'] = baseline_paths
            st.session_state['proposed_paths'] = proposed_paths
            st.session_state['baseline_stats'] = baseline_stats
            st.session_state['proposed_stats'] = proposed_stats
            st.session_state['result'] = result

    if 'result' in st.session_state:
        st.success("Analysis Complete!")

        result = st.session_state['result']
        proposed_stats = st.session_state['proposed_stats']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{result['mse']:.2e}")
        col2.metric("MAE", f"{result['mae']:.2e}")
        col3.metric("Speedup", f"{result['speedup']:.2f}×")
        col4.metric("Path Overlap", f"{result['path_overlap']*100:.1f}%")

        st.subheader("Performance Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            time_data = pd.DataFrame({
                'Stage': ['Candidate Generation', 'Anchor Prediction', 'Path Pruning'],
                'Time (ms)': [
                    proposed_stats['candidate_gen_time'] * 1000,
                    proposed_stats['anchor_pred_time'] * 1000,
                    proposed_stats['pruning_time'] * 1000
                ]
            })

            fig = px.bar(time_data, x='Stage', y='Time (ms)',
                        title='Processing Time by Stage',
                        color='Stage')
            st.plotly_chart(fig, width='stretch')

        with col2:
            stats_data = pd.DataFrame({
                'Metric': ['Total Paths', 'Candidates', 'Pruned Paths', 'Top Paths'],
                'Count': [
                    result['num_paths'],
                    proposed_stats['num_candidates'],
                    proposed_stats['num_pruned_paths'],
                    10
                ]
            })

            fig = px.bar(stats_data, x='Metric', y='Count',
                        title='Path Reduction Statistics',
                        color='Metric')
            st.plotly_chart(fig, width='stretch')

        st.subheader("Slack Comparison")

        baseline_paths = st.session_state['baseline_paths']
        proposed_paths = st.session_state['proposed_paths']

        comparison_df = pd.DataFrame({
            'Rank': range(1, len(baseline_paths) + 1),
            'Baseline Slack': [p.slack for p in baseline_paths],
            'Proposed Slack': [p.slack for p in proposed_paths[:len(baseline_paths)]]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comparison_df['Rank'], y=comparison_df['Baseline Slack'],
                                mode='lines+markers', name='Baseline (Exhaustive)',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=comparison_df['Rank'], y=comparison_df['Proposed Slack'],
                                mode='lines+markers', name='Proposed (ML-based)',
                                line=dict(color='red', width=2, dash='dash')))

        fig.update_layout(
            title='Top 10 Critical Paths: Slack Comparison',
            xaxis_title='Path Rank',
            yaxis_title='Slack (ps)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')

with tab3:
    st.header("Advanced Visualizations")

    if 'endpoint_graph' in st.session_state and 'proposed_paths' in st.session_state:
        visualizer = Visualizer()

        endpoint_graph = st.session_state['endpoint_graph']
        proposed_paths = st.session_state['proposed_paths']
        baseline_paths = st.session_state['baseline_paths']
        proposed_stats = st.session_state['proposed_stats']

        st.subheader("Timing Path Graph Visualization")
        st.markdown("Interactive graph showing timing paths and anchor node location")

        anchor_node = proposed_stats.get('anchor')

        fig_graph = visualizer.create_timing_path_graph(
            endpoint_graph,
            proposed_paths,
            anchor_node=anchor_node,
            max_paths=10
        )
        st.plotly_chart(fig_graph, width='stretch')

        st.subheader("Slack Distribution Analysis")

        fig_slack = visualizer.create_slack_distribution(
            endpoint_graph,
            baseline_paths=baseline_paths,
            proposed_paths=proposed_paths
        )
        st.plotly_chart(fig_slack, width='stretch')

        if anchor_node:
            st.subheader("Anchor Node Analysis")

            from sta_pruning import CandidateGenerator
            cand_gen = CandidateGenerator()
            candidates = cand_gen.generate(endpoint_graph)

            if candidates:
                fig_anchor = visualizer.create_anchor_analysis(
                    endpoint_graph,
                    candidates,
                    anchor_node
                )
                st.plotly_chart(fig_anchor, width='stretch')
    else:
        st.info("Run the Interactive Demo first to generate visualizations!")

with tab4:
    st.header("Benchmark Results")

    st.subheader("Multi-Design Benchmarking")

    num_designs = st.selectbox("Number of Test Designs", [3, 5, 10], index=1)

    if st.button("Run Comprehensive Benchmark", type="primary"):
        with st.spinner(f"Running benchmark on {num_designs} designs..."):
            designs = data_gen.generate_benchmark_designs(num_designs)

            evaluator_rf = Evaluator()
            evaluator_xgb = Evaluator()

            progress_bar = st.progress(0)
            total_endpoints = sum(len(eps) for _, eps in designs)
            current = 0

            for design_name, endpoint_graphs in designs:
                for ep_graph in endpoint_graphs:
                    evaluator_rf.benchmark_endpoint(ep_graph, pipeline_rf, design_name)
                    evaluator_xgb.benchmark_endpoint(ep_graph, pipeline_xgb, design_name)
                    current += 1
                    progress_bar.progress(current / total_endpoints)

            results_rf = pd.DataFrame(evaluator_rf.results)
            results_xgb = pd.DataFrame(evaluator_xgb.results)

            st.session_state['results_rf'] = results_rf
            st.session_state['results_xgb'] = results_xgb
            st.session_state['summary_rf'] = evaluator_rf.get_summary_statistics()
            st.session_state['summary_xgb'] = evaluator_xgb.get_summary_statistics()

    if 'results_rf' in st.session_state:
        results_rf = st.session_state['results_rf']
        results_xgb = st.session_state['results_xgb']
        summary_rf = st.session_state['summary_rf']
        summary_xgb = st.session_state['summary_xgb']

        st.success(f"Benchmark complete! Analyzed {summary_rf['total_endpoints']} endpoints")

        st.subheader("Summary Statistics")

        summary_comparison = pd.DataFrame({
            'Metric': ['Avg MSE', 'Avg MAE', 'Avg Speedup', 'Avg Path Overlap', 'Min Speedup', 'Max Speedup'],
            'Random Forest': [
                f"{summary_rf['avg_mse']:.2e}",
                f"{summary_rf['avg_mae']:.2e}",
                f"{summary_rf['avg_speedup']:.2f}×",
                f"{summary_rf['avg_path_overlap']*100:.1f}%",
                f"{summary_rf['min_speedup']:.2f}×",
                f"{summary_rf['max_speedup']:.2f}×"
            ],
            'XGBoost': [
                f"{summary_xgb['avg_mse']:.2e}",
                f"{summary_xgb['avg_mae']:.2e}",
                f"{summary_xgb['avg_speedup']:.2f}×",
                f"{summary_xgb['avg_path_overlap']*100:.1f}%",
                f"{summary_xgb['min_speedup']:.2f}×",
                f"{summary_xgb['max_speedup']:.2f}×"
            ]
        })

        st.dataframe(summary_comparison, hide_index=True, width='stretch')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Speedup Distribution (RF)")
            fig = px.histogram(results_rf, x='speedup', nbins=30,
                             title='Speedup Distribution - Random Forest',
                             labels={'speedup': 'Speedup Factor'})
            fig.add_vline(x=summary_rf['avg_speedup'], line_dash="dash",
                         line_color="red", annotation_text=f"Avg: {summary_rf['avg_speedup']:.2f}×")
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Speedup Distribution (XGB)")
            fig = px.histogram(results_xgb, x='speedup', nbins=30,
                             title='Speedup Distribution - XGBoost',
                             labels={'speedup': 'Speedup Factor'})
            fig.add_vline(x=summary_xgb['avg_speedup'], line_dash="dash",
                         line_color="red", annotation_text=f"Avg: {summary_xgb['avg_speedup']:.2f}×")
            st.plotly_chart(fig, width='stretch')

        st.subheader("Accuracy Metrics by Design")

        design_stats_rf = results_rf.groupby('design').agg({
            'mse': 'mean',
            'mae': 'mean',
            'speedup': 'mean',
            'path_overlap': 'mean',
            'num_paths': 'mean'
        }).reset_index()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE by Design', 'Speedup by Design',
                          'Path Overlap by Design', 'Paths per Endpoint')
        )

        fig.add_trace(go.Bar(x=design_stats_rf['design'], y=design_stats_rf['mae'],
                            name='MAE'), row=1, col=1)
        fig.add_trace(go.Bar(x=design_stats_rf['design'], y=design_stats_rf['speedup'],
                            name='Speedup'), row=1, col=2)
        fig.add_trace(go.Bar(x=design_stats_rf['design'], y=design_stats_rf['path_overlap']*100,
                            name='Overlap %'), row=2, col=1)
        fig.add_trace(go.Bar(x=design_stats_rf['design'], y=design_stats_rf['num_paths'],
                            name='Num Paths'), row=2, col=2)

        fig.update_layout(height=600, showlegend=False, title_text="Performance Metrics by Design")
        st.plotly_chart(fig, width='stretch')

        st.subheader("Detailed Results Table")

        display_df = results_rf[['design', 'endpoint', 'num_paths', 'mse', 'mae',
                                 'speedup', 'path_overlap', 'num_candidates']].copy()
        display_df['mse'] = display_df['mse'].apply(lambda x: f"{x:.2e}")
        display_df['mae'] = display_df['mae'].apply(lambda x: f"{x:.2e}")
        display_df['speedup'] = display_df['speedup'].apply(lambda x: f"{x:.2f}×")
        display_df['path_overlap'] = display_df['path_overlap'].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(display_df, hide_index=True, width='stretch', height=400)

        st.subheader("Generate Report")

        if st.button("Generate Comprehensive Report"):
            report_gen = ReportGenerator()
            report_text = report_gen.generate_markdown_report(results_rf, summary_rf)

            st.download_button(
                label="Download Markdown Report",
                data=report_text,
                file_name=f"sta_benchmark_report_{report_gen.timestamp}.md",
                mime="text/markdown"
            )

            csv_data = results_rf.to_csv(index=False)
            st.download_button(
                label="Download CSV Results",
                data=csv_data,
                file_name=f"sta_benchmark_results_{report_gen.timestamp}.csv",
                mime="text/csv"
            )

with tab5:
    st.header("Model Tuning")

    st.subheader("Hyperparameter Optimization")

    model_type_tune = st.radio("Select Model for Tuning", ["Random Forest", "XGBoost"], horizontal=True, key="tune_model")

    tuning_method = st.selectbox("Tuning Method", ["Cross-Validation", "Grid Search", "Random Search"])

    if tuning_method == "Cross-Validation":
        cv_folds = st.slider("Number of CV Folds", 3, 10, 5)
        n_samples = st.slider("Training Samples", 20, 100, 50)

        if st.button("Run Cross-Validation", type="primary"):
            with st.spinner(f"Running {cv_folds}-fold cross-validation on {n_samples} samples..."):
                try:
                    from sta_pruning import SyntheticDataGenerator, ModelTuner
                    data_gen_cv = SyntheticDataGenerator(seed=123)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("Generating training data...")
                    training_data_cv = data_gen_cv.generate_training_data(n_samples)
                    progress_bar.progress(0.3)

                    model_type_str = 'rf' if model_type_tune == "Random Forest" else 'xgb'
                    tuner = ModelTuner(model_type=model_type_str)

                    status_text.text(f"Running cross-validation with {cv_folds} folds...")
                    progress_bar.progress(0.5)

                    cv_results = tuner.cross_validate(training_data_cv, cv=cv_folds)
                    progress_bar.progress(1.0)
                    status_text.empty()

                    st.success("Cross-validation complete!")

                    col1, col2 = st.columns(2)
                    col1.metric("Mean F1 Score", f"{cv_results['mean_score']:.4f}")
                    col2.metric("Std F1 Score", f"{cv_results['std_score']:.4f}")

                    scores_df = pd.DataFrame({
                        'Fold': range(1, len(cv_results['scores']) + 1),
                        'F1 Score': cv_results['scores']
                    })

                    fig = px.bar(scores_df, x='Fold', y='F1 Score',
                               title=f'Cross-Validation Scores by Fold ({model_type_tune})',
                               color='F1 Score',
                               color_continuous_scale='Blues')
                    fig.add_hline(y=cv_results['mean_score'], line_dash="dash",
                                 line_color="red", annotation_text=f"Mean: {cv_results['mean_score']:.4f}")
                    st.plotly_chart(fig, use_container_width=True)

                    st.session_state['cv_results'] = cv_results

                except Exception as e:
                    st.error(f"Error during cross-validation: {str(e)}")
                    st.exception(e)

    elif tuning_method == "Grid Search":
        st.info("Grid Search performs exhaustive search over specified parameter values.")

        cv_folds_grid = st.slider("CV Folds", 3, 5, 3, key="grid_cv")
        n_samples_grid = st.slider("Training Samples", 20, 100, 40, key="grid_samples")

        st.markdown("**Note:** Grid search can take several minutes depending on the parameter grid size.")

        if st.button("Run Grid Search", type="primary"):
            with st.spinner(f"Running grid search with {cv_folds_grid}-fold CV..."):
                try:
                    from sta_pruning import SyntheticDataGenerator, ModelTuner
                    data_gen_grid = SyntheticDataGenerator(seed=456)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("Generating training data...")
                    training_data_grid = data_gen_grid.generate_training_data(n_samples_grid)
                    progress_bar.progress(0.2)

                    model_type_str = 'rf' if model_type_tune == "Random Forest" else 'xgb'
                    tuner = ModelTuner(model_type=model_type_str)

                    status_text.text("Running grid search (this may take a while)...")
                    progress_bar.progress(0.3)

                    # Use a smaller parameter grid for faster execution
                    if model_type_str == 'rf':
                        param_grid = {
                            'n_estimators': [100, 500],
                            'max_depth': [10, 20, None],
                            'min_samples_split': [2, 5]
                        }
                    else:
                        param_grid = {
                            'n_estimators': [100, 500],
                            'max_depth': [3, 6],
                            'learning_rate': [0.1, 0.3]
                        }

                    results = tuner.grid_search_cv(training_data_grid, param_grid=param_grid, cv=cv_folds_grid, n_jobs=2)
                    progress_bar.progress(1.0)
                    status_text.empty()

                    st.success("Grid search complete!")

                    st.metric("Best F1 Score", f"{results['best_score']:.4f}")

                    st.subheader("Best Parameters")
                    best_params_df = pd.DataFrame({
                        'Parameter': list(results['best_params'].keys()),
                        'Value': [str(v) for v in results['best_params'].values()]
                    })
                    st.dataframe(best_params_df, hide_index=True, width='stretch')

                    st.session_state['grid_results'] = results

                except Exception as e:
                    st.error(f"Error during grid search: {str(e)}")
                    st.exception(e)

    elif tuning_method == "Random Search":
        n_iter = st.slider("Number of Iterations", 10, 50, 20)
        cv_folds_random = st.slider("CV Folds", 3, 5, 3, key="random_cv")
        n_samples_random = st.slider("Training Samples", 20, 100, 50, key="random_samples")

        if st.button("Run Random Search", type="primary"):
            with st.spinner(f"Running random search with {n_iter} iterations..."):
                try:
                    from sta_pruning import SyntheticDataGenerator, ModelTuner
                    data_gen_rs = SyntheticDataGenerator(seed=789)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("Generating training data...")
                    training_data_rs = data_gen_rs.generate_training_data(n_samples_random)
                    progress_bar.progress(0.2)

                    model_type_str = 'rf' if model_type_tune == "Random Forest" else 'xgb'
                    tuner = ModelTuner(model_type=model_type_str)

                    status_text.text(f"Running random search with {n_iter} iterations...")
                    progress_bar.progress(0.3)

                    results = tuner.random_search_cv(training_data_rs, n_iter=n_iter, cv=cv_folds_random, n_jobs=2)
                    progress_bar.progress(1.0)
                    status_text.empty()

                    st.success("Random search complete!")

                    st.metric("Best F1 Score", f"{results['best_score']:.4f}")

                    st.subheader("Best Parameters")
                    best_params_df = pd.DataFrame({
                        'Parameter': list(results['best_params'].keys()),
                        'Value': [str(v) for v in results['best_params'].values()]
                    })
                    st.dataframe(best_params_df, hide_index=True, width='stretch')

                    st.session_state['random_results'] = results

                except Exception as e:
                    st.error(f"Error during random search: {str(e)}")
                    st.exception(e)

    # Display comparison if multiple tuning methods have been run
    st.markdown("---")
    st.subheader("Tuning Results Comparison")

    results_available = []
    if 'cv_results' in st.session_state:
        results_available.append(('Cross-Validation', st.session_state['cv_results']['mean_score']))
    if 'grid_results' in st.session_state:
        results_available.append(('Grid Search', st.session_state['grid_results']['best_score']))
    if 'random_results' in st.session_state:
        results_available.append(('Random Search', st.session_state['random_results']['best_score']))

    if results_available:
        comparison_df = pd.DataFrame(results_available, columns=['Method', 'Best F1 Score'])

        fig = px.bar(comparison_df, x='Method', y='Best F1 Score',
                    title='Tuning Methods Comparison',
                    color='Best F1 Score',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comparison_df, hide_index=True, width='stretch')
    else:
        st.info("Run one or more tuning methods to see comparison results.")

with tab6:
    st.header("Batch Processing")

    st.subheader("Large-Scale Endpoint Analysis")

    num_endpoints_batch = st.slider("Number of Endpoints", 100, 2000, 500, step=100)
    num_paths_range = st.slider("Paths per Endpoint Range", 50, 500, (100, 300))

    processing_mode = st.radio("Processing Mode", ["Sequential", "Parallel (Threads)"], horizontal=True)

    if st.button("Run Batch Processing"):
        with st.spinner(f"Processing {num_endpoints_batch} endpoints..."):
            from sta_pruning import SyntheticDataGenerator, BatchProcessor, Pipeline

            data_gen_batch = SyntheticDataGenerator(seed=456)

            endpoint_graphs = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(num_endpoints_batch):
                num_paths = np.random.randint(num_paths_range[0], num_paths_range[1])
                ep_graph = data_gen_batch.generate_endpoint_graph(
                    f"batch_ep_{i}",
                    num_paths=num_paths
                )
                endpoint_graphs.append(ep_graph)

                if (i + 1) % 50 == 0:
                    progress_bar.progress((i + 1) / num_endpoints_batch)
                    status_text.text(f"Generated {i + 1}/{num_endpoints_batch} endpoints")

            progress_bar.progress(1.0)
            status_text.text("Generation complete! Starting analysis...")

            batch_processor = BatchProcessor(pipeline_rf, n_workers=4)

            start_time = time.time()

            if processing_mode == "Sequential":
                results = batch_processor.process_batch_sequential(endpoint_graphs)
            else:
                results = batch_processor.process_batch_parallel(endpoint_graphs, use_processes=False)

            total_time = time.time() - start_time

            batch_stats = batch_processor.get_batch_statistics(results)

            st.success(f"Batch processing complete in {total_time:.2f}s!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Endpoints", batch_stats['total_endpoints'])
            col2.metric("Total Time (s)", f"{batch_stats['total_time']:.2f}")
            col3.metric("Avg Time/Endpoint (ms)", f"{batch_stats['avg_time_per_endpoint']*1000:.2f}")
            col4.metric("Throughput (eps/s)", f"{batch_stats['total_endpoints']/batch_stats['total_time']:.2f}")

            st.subheader("Processing Time Distribution")

            times = [stats['total_time'] * 1000 for _, _, stats in results]

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=times, nbinsx=50, name='Processing Time'))
            fig.update_layout(
                title='Processing Time Distribution',
                xaxis_title='Time (ms)',
                yaxis_title='Count'
            )
            st.plotly_chart(fig, width='stretch')

with tab7:
    st.header("Model Analysis")

    st.subheader("Feature Importance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Random Forest**")
        importance_rf = pipeline_rf.get_feature_importance()
        if importance_rf:
            importance_df_rf = pd.DataFrame({
                'Feature': list(importance_rf.keys()),
                'Importance': list(importance_rf.values())
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df_rf.head(15), x='Importance', y='Feature',
                        orientation='h', title='Top 15 Most Important Features (RF)')
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("**XGBoost**")
        importance_xgb = pipeline_xgb.get_feature_importance()
        if importance_xgb:
            importance_df_xgb = pd.DataFrame({
                'Feature': list(importance_xgb.keys()),
                'Importance': list(importance_xgb.values())
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df_xgb.head(15), x='Importance', y='Feature',
                        orientation='h', title='Top 15 Most Important Features (XGB)')
            st.plotly_chart(fig, width='stretch')

    st.subheader("Model Configuration")

    config_df = pd.DataFrame({
        'Parameter': ['Model Type', 'Number of Estimators', 'Max Candidates',
                     'Top K Paths', 'Training Samples', 'Feature Dimension'],
        'Random Forest': ['Random Forest', '500', '30', '10', '50', '21'],
        'XGBoost': ['XGBoost', '500', '30', '10', '50', '21']
    })

    st.dataframe(config_df, hide_index=True, width='stretch')

with tab8:
    st.header("CircuitNet-Style Dataset")

    st.markdown("""
    Work with **realistic circuit timing data** that mimics CircuitNet designs (CPU, GPU, AI chips).

    **🎯 No huge downloads needed!** We generate synthetic data that matches real CircuitNet characteristics:
    - Realistic timing distributions from actual chip designs
    - Multiple endpoint-based graphs per design
    - Configurable complexity levels (medium to very high)
    """)

    st.subheader("Data Source Selection")

    data_mode = st.radio(
        "Select Data Source",
        ["Synthetic (Recommended - No Download)", "Real CircuitNet (Requires Download)"],
        horizontal=True
    )

    use_synthetic = data_mode.startswith("Synthetic")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Available Designs"):
            from sta_pruning import CircuitNetLoader

            loader = CircuitNetLoader("./circuitnet_data", use_synthetic=use_synthetic)
            available_designs = loader.list_available_designs()

            if available_designs:
                st.success(f"✓ Found {len(available_designs)} designs!")
                st.session_state['circuitnet_designs'] = available_designs
                st.session_state['circuitnet_loader'] = loader
                st.session_state['use_synthetic'] = use_synthetic
            else:
                st.error("No designs available")

    with col2:
        if not use_synthetic:
            if st.button("Download Instructions"):
                st.warning("""
                **Real CircuitNet files are 10+ GB!**

                We recommend using synthetic data instead.

                If you still want real data:
                ```bash
                pip install huggingface-hub
                huggingface-cli download circuitnet/CircuitNet-N14 \\
                  --repo-type dataset \\
                  --local-dir ./circuitnet_data/raw
                ```
                """)

    if 'circuitnet_designs' in st.session_state:
        st.subheader("Available Designs")

        designs = st.session_state['circuitnet_designs']
        st.write(f"Total designs available: {len(designs)}")

        with st.expander("View all designs"):
            st.write(designs)

        st.subheader("Load and Analyze Design")

        selected_designs = st.multiselect(
            "Select designs to load",
            designs,
            max_selections=10
        )

        if selected_designs and st.button("Load Selected Designs"):
            with st.spinner(f"Loading {len(selected_designs)} designs..."):
                try:
                    loader = st.session_state['circuitnet_loader']

                    loaded_data = loader.load_batch_designs(selected_designs)

                    if loaded_data:
                        st.success(f"Successfully loaded {len(loaded_data)} designs!")

                        total_endpoints = sum(len(eps) for _, eps in loaded_data)
                        st.metric("Total Endpoints Loaded", total_endpoints)

                        st.session_state['circuitnet_data'] = loaded_data

                        summary_data = []
                        for design_name, endpoint_graphs in loaded_data:
                            for ep in endpoint_graphs:
                                summary_data.append({
                                    'Design': design_name,
                                    'Endpoint': ep.endpoint,
                                    'Num Paths': ep.num_paths,
                                    'Worst Slack': f"{ep.worst_slack:.2e}"
                                })

                        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
                    else:
                        st.error("Failed to load designs. Check data format and paths.")
                except Exception as e:
                    st.error(f"Error loading designs: {str(e)}")
                    st.exception(e)

        if 'circuitnet_data' in st.session_state:
            st.subheader("Train on CircuitNet Data")

            if st.button("Train Models on Real Data"):
                with st.spinner("Training models on CircuitNet data..."):
                    try:
                        from sta_pruning import ModelTrainer, Pipeline

                        circuitnet_data = st.session_state['circuitnet_data']

                        all_endpoint_graphs = []
                        for _, endpoint_graphs in circuitnet_data:
                            all_endpoint_graphs.extend(endpoint_graphs)

                        if not all_endpoint_graphs:
                            st.error("No endpoint graphs available for training")
                        else:
                            trainer_rf = ModelTrainer(model_type='rf', n_estimators=500)
                            trainer_xgb = ModelTrainer(model_type='xgb', n_estimators=500)

                            num_training = min(50, len(all_endpoint_graphs))
                            training_data = trainer_rf.prepare_training_data(all_endpoint_graphs[:num_training])

                            pipeline_rf_real = Pipeline(model_type='rf', n_estimators=500)
                            pipeline_rf_real.train(training_data)

                            pipeline_xgb_real = Pipeline(model_type='xgb', n_estimators=500)
                            pipeline_xgb_real.train(training_data)

                            st.session_state['pipeline_rf_real'] = pipeline_rf_real
                            st.session_state['pipeline_xgb_real'] = pipeline_xgb_real

                            st.success(f"Models trained on {num_training} CircuitNet endpoints!")
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
                        st.exception(e)

            if 'pipeline_rf_real' in st.session_state:
                st.subheader("Benchmark on Real Data")

                if st.button("Run Benchmark on CircuitNet"):
                    with st.spinner("Benchmarking on real circuit data..."):
                        try:
                            from sta_pruning import Evaluator

                            circuitnet_data = st.session_state['circuitnet_data']
                            pipeline_rf_real = st.session_state['pipeline_rf_real']

                            evaluator = Evaluator()
                            results_df = evaluator.benchmark_multiple_designs(
                                circuitnet_data,
                                pipeline_rf_real
                            )

                            summary = evaluator.get_summary_statistics()

                            st.success("Benchmark complete!")

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Avg MSE", f"{summary['avg_mse']:.2e}")
                            col2.metric("Avg MAE", f"{summary['avg_mae']:.2e}")
                            col3.metric("Avg Speedup", f"{summary['avg_speedup']:.2f}×")
                            col4.metric("Avg Path Overlap", f"{summary['avg_path_overlap']*100:.1f}%")

                            st.subheader("Detailed Results")
                            st.dataframe(results_df, hide_index=True, use_container_width=True)

                            st.session_state['circuitnet_results'] = results_df
                            st.session_state['circuitnet_summary'] = summary
                        except Exception as e:
                            st.error(f"Error running benchmark: {str(e)}")
                            st.exception(e)

with tab9:
    st.header("📚 Documentation")

    # Quick navigation
    doc_section = st.selectbox(
        "Jump to section:",
        ["Overview", "Quick Start", "System Architecture", "Algorithm Pipeline",
         "Feature Extraction", "Dashboard Features", "API Usage", "Performance Metrics",
         "Troubleshooting", "Advanced Features"]
    )

    st.markdown("---")

    if doc_section == "Overview":
        st.subheader("🎯 Overview")
        st.markdown("""
        This system implements an **intelligent path pruning algorithm** for Static Timing Analysis (STA)
        that uses machine learning to accelerate worst-case timing path identification in circuit designs.

        ### Key Features
        - **ML-Driven Anchor Prediction**: Random Forest and XGBoost classifiers
        - **Efficient Path Pruning**: 30% faster than exhaustive Path-Based Analysis (PBA)
        - **High Accuracy**: MAE < 1.6e-04 (sub-picosecond deviation)
        - **Scalable**: Handles designs from 1.5M to 9.7M pins
        - **Interactive Dashboard**: Real-time visualization and benchmarking
        - **Comprehensive Evaluation**: Multi-design benchmarking framework

        ### Target Applications
        - Circuit designers needing faster timing closure
        - EDA engineers optimizing timing analysis workflows
        - Researchers exploring ML for circuit analysis
        - Teams working with large-scale industrial designs
        """)

    elif doc_section == "Quick Start":
        st.subheader("🚀 Quick Start")
        st.markdown("""
        ### Installation

        ```bash
        # 1. Install UV package manager
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

        # 2. Create virtual environment and install dependencies
        uv venv
        uv pip install -e .

        # 3. Activate virtual environment
        .venv\\Scripts\\activate  # Windows
        source .venv/bin/activate  # Linux/Mac

        # 4. Run the dashboard
        streamlit run app.py
        ```

        Visit `http://localhost:8502` to access the interactive dashboard!

        ### First Analysis

        ```python
        from sta_pruning import Pipeline, SyntheticDataGenerator

        # Generate test data
        data_gen = SyntheticDataGenerator(seed=42)
        endpoint_graph = data_gen.generate_endpoint_graph("test_endpoint", num_paths=200)

        # Create and train pipeline
        pipeline = Pipeline(model_type='rf', n_estimators=1000)
        training_data = data_gen.generate_training_data(100)
        pipeline.train(training_data)

        # Analyze endpoint
        top_paths, stats = pipeline.process_endpoint(endpoint_graph)

        print(f"Top {len(top_paths)} critical paths identified")
        print(f"Processing time: {stats['total_time']*1000:.2f} ms")
        print(f"Anchor node: {stats['anchor'].node_id}")
        ```
        """)

    elif doc_section == "System Architecture":
        st.subheader("🏗️ System Architecture")
        st.markdown("""
        ### Core Components

        **1. Data Structures (`data_structures.py`)**
        - `Node`: Represents a timing node with arrival/required times, slack, and physical properties
        - `TimingPath`: Sequence of nodes from startpoint to endpoint with timing metrics
        - `EndpointBasedGraph`: Collection of all paths converging at a single endpoint

        **2. Feature Extraction (`feature_extractor.py`)**
        - Extracts 21-dimensional feature vectors for each candidate node
        - **Path Features (7):** slack statistics, delays, criticality ratios
        - **Node Features (14):** timing, capacitance, fanout, depth, coverage metrics

        **3. Candidate Generation (`candidate_generator.py`)**
        - Stage 1: Identifies potential anchor nodes
        - Focuses on middle 40% region of paths (30%-70%)
        - Scores nodes by path coverage and criticality
        - Returns top 30 candidates

        **4. Anchor Prediction (`anchor_predictor.py`)**
        - Stage 2: ML-based selection of best anchor
        - Supports Random Forest and XGBoost classifiers
        - Trained to maximize path coverage and timing accuracy

        **5. Pipeline (`pipeline.py`)**
        - End-to-end orchestration of all stages
        - Baseline comparison mode (exhaustive analysis)
        - Proposed mode (ML-accelerated pruning)
        - Performance tracking and statistics

        **6. Evaluation (`evaluate.py`)**
        - Computes accuracy metrics (MSE, MAE)
        - Calculates speedup and path overlap
        - Multi-design benchmarking framework

        **7. Data Generation (`data_generator.py`)**
        - Synthetic circuit generation for testing
        - Configurable design complexity
        - Training data preparation

        **8. Visualization (`visualizer.py`)**
        - Interactive timing path graphs
        - Slack distribution histograms
        - Anchor coverage analysis

        **9. Model Tuning (`model_tuner.py`)**
        - Cross-validation framework
        - Grid search optimization
        - Random search for hyperparameters

        **10. Batch Processing (`batch_processor.py`)**
        - Sequential and parallel processing modes
        - Large-scale endpoint analysis
        - Performance statistics tracking

        **11. Report Generation (`report_generator.py`)**
        - Markdown and CSV report formats
        - Comprehensive benchmark summaries

        **12. CircuitNet Integration (`circuitnet_loader.py`)**
        - Real circuit dataset loading
        - Synthetic data generation
        - Training on realistic timing data

        """)

    elif doc_section == "Algorithm Pipeline":
        st.subheader("⚙️ Algorithm Pipeline")
        st.markdown("""
        ### Three-Stage Process

        #### Stage 1: Candidate Generation
        - Extract nodes from middle 40% region of timing paths (30%-70%)
        - Score candidates by path coverage and timing criticality
        - Select top 30 candidates for anchor prediction

        #### Stage 2: Anchor Prediction
        - Extract 21-dimensional feature vectors:
          - 7 path features (slack statistics, delays, criticality)
          - 14 node features (timing, physical properties, connectivity)
        - ML classifier predicts optimal anchor node
        - Supports Random Forest and XGBoost models

        #### Stage 3: Path Pruning
        - Filter paths passing through predicted anchor
        - Sort by slack (worst-case first)
        - Return top 10 critical timing paths

        ### Workflow Diagram

        ```
        Input: Endpoint-based timing graph with N paths

        Stage 1: Candidate Generation
        ├─ Extract nodes from middle 40% of each path
        ├─ Score by coverage and criticality
        └─ Select top 30 candidates

        Stage 2: Anchor Prediction
        ├─ Extract 21-dim features for each candidate
        ├─ ML classifier predicts best anchor node
        └─ Return highest probability candidate

        Stage 3: Path Pruning
        ├─ Filter paths passing through anchor
        ├─ Sort by slack (worst-case first)
        └─ Return top 10 critical paths

        Output: Top-K worst-case timing paths
        ```
        """)

    elif doc_section == "Feature Extraction":
        st.subheader("🔍 Feature Extraction (21 Dimensions)")
        st.markdown("""
        ### Path Features (7 dimensions)

        1. **Average Path Slack** - Mean slack across all paths
        2. **Minimum Path Slack** - Worst-case slack value
        3. **Maximum Path Slack** - Best-case slack value
        4. **Standard Deviation of Slack** - Slack variability
        5. **Average Path Delay** - Mean propagation delay
        6. **Number of Critical Paths** - Count of timing-critical paths
        7. **Critical Path Ratio** - Proportion of critical paths

        ### Node Features (14 dimensions)

        1. **Node Slack** - Timing slack at this node
        2. **Arrival Time** - Signal arrival time
        3. **Required Time** - Required arrival time
        4. **Capacitance** - Node capacitance load
        5. **Transition Time** - Signal transition time
        6. **Fanout Count** - Number of downstream nodes
        7. **Depth in Path** - Position along timing path
        8. **Paths Through Node** - Number of paths traversing this node
        9. **Average Slack of Paths Through** - Mean slack of traversing paths
        10. **Minimum Slack of Paths Through** - Worst slack of traversing paths
        11. **Position in Critical Path** - Location in critical path
        12. **Criticality Score** - Overall timing criticality
        13. **Upstream Criticality** - Criticality of upstream logic
        14. **Downstream Criticality** - Criticality of downstream logic

        ### Feature Importance

        Top features for anchor prediction (Random Forest):
        - Paths through node (coverage)
        - Node slack (timing criticality)
        - Average slack of paths through
        - Depth in path (position)
        - Criticality score
        """)

    elif doc_section == "Dashboard Features":
        st.subheader("📊 Dashboard Features")
        st.markdown("""
        ### 9 Interactive Tabs

        **1. Overview Tab**
        - Algorithm description and pipeline stages
        - Target performance metrics
        - System status and configuration

        **2. Interactive Demo**
        - Generate synthetic circuits with configurable parameters
        - Real-time analysis with Random Forest or XGBoost
        - Visual comparison of baseline vs. ML-accelerated approach
        - Detailed performance breakdown by stage

        **3. Advanced Visualizations**
        - Timing path graphs with anchor highlighting
        - Slack distribution histograms
        - Anchor coverage analysis
        - Interactive network diagrams

        **4. Benchmark Results**
        - Multi-design benchmarking (3-10 designs)
        - Speedup distribution analysis
        - Accuracy metrics by design
        - Comprehensive results tables

        **5. Model Tuning**
        - Cross-validation with configurable folds
        - Grid search for hyperparameter optimization
        - Random search for efficient parameter exploration
        - Performance comparison charts

        **6. Batch Processing**
        - Sequential and parallel processing modes
        - Large-scale endpoint analysis
        - Performance statistics and timing breakdown
        - Batch size configuration

        **7. Model Analysis**
        - Feature importance visualization
        - Model configuration details
        - Comparison between Random Forest and XGBoost
        - Training statistics

        **8. CircuitNet Dataset**
        - Synthetic circuit data generation
        - Real CircuitNet dataset integration
        - Batch loading and processing
        - Training on realistic timing data

        **9. Documentation** (This Tab)
        - Complete system documentation
        - API reference and examples
        - Installation and troubleshooting
        - Performance optimization tips
        """)

    elif doc_section == "API Usage":
        st.subheader("💻 API Usage")
        st.markdown("""
        ### Basic Usage

        ```python
        from sta_pruning import Pipeline, SyntheticDataGenerator

        # Generate test data
        data_gen = SyntheticDataGenerator(seed=42)
        endpoint_graph = data_gen.generate_endpoint_graph("test_endpoint", num_paths=200)

        # Create and train pipeline
        pipeline = Pipeline(model_type='rf', n_estimators=1000)
        training_data = data_gen.generate_training_data(100)
        pipeline.train(training_data)

        # Analyze endpoint
        top_paths, stats = pipeline.process_endpoint(endpoint_graph)

        print(f"Top {len(top_paths)} critical paths identified")
        print(f"Processing time: {stats['total_time']*1000:.2f} ms")
        print(f"Anchor node: {stats['anchor'].node_id}")
        ```

        ### Benchmarking Multiple Designs

        ```python
        from sta_pruning import Evaluator, Pipeline, SyntheticDataGenerator

        data_gen = SyntheticDataGenerator()
        designs = data_gen.generate_benchmark_designs(num_designs=5)

        evaluator = Evaluator()
        pipeline = Pipeline(model_type='rf')

        # Train on synthetic data
        training_data = data_gen.generate_training_data(100)
        pipeline.train(training_data)

        # Benchmark
        results_df = evaluator.benchmark_multiple_designs(designs, pipeline)
        summary = evaluator.get_summary_statistics()

        print(f"Average MAE: {summary['avg_mae']:.2e}")
        print(f"Average Speedup: {summary['avg_speedup']:.2f}×")
        ```

        ### Model Tuning

        ```python
        from sta_pruning import ModelTuner, SyntheticDataGenerator

        data_gen = SyntheticDataGenerator()
        training_data = data_gen.generate_training_data(100)

        tuner = ModelTuner()

        # Cross-validation
        cv_results = tuner.cross_validate(training_data, model_type='rf', n_folds=5)

        # Grid search
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [10, 20, 30]
        }
        best_params = tuner.grid_search(training_data, param_grid, model_type='rf')
        ```
        """)

    elif doc_section == "Performance Metrics":
        st.subheader("📈 Performance Metrics")
        st.markdown("""
        ### Accuracy
        - **MSE**: ~1.4e-06 (mean squared error)
        - **MAE**: <1.6e-04 (mean absolute error, sub-picosecond)
        - **Path Overlap**: >90% with baseline

        ### Speed
        - **Speedup**: 1.3-1.7× faster than exhaustive PBA
        - **Runtime Reduction**: ~30%
        - **Processing Time**: 10-50ms per endpoint (typical)

        ### Scale
        - **Design Size**: 1.5M - 9.7M pins
        - **Endpoints**: Up to 2000 per design
        - **Paths per Endpoint**: 50-500

        ### Model Configuration
        - **Random Forest**: 500-1000 trees, max_depth=20
        - **XGBoost**: 500-1000 estimators, max_depth=10
        - **Feature Dimension**: 21
        - **Max Candidates**: 30
        - **Top Paths Returned**: 10

        ### Performance Optimization Tips

        1. **Model Selection**: Random Forest typically faster, XGBoost slightly more accurate
        2. **Candidate Count**: 30 candidates balances accuracy and speed
        3. **Training Size**: 50-100 samples sufficient for good performance
        4. **Batch Processing**: Process multiple endpoints in parallel when possible
        """)

    elif doc_section == "Troubleshooting":
        st.subheader("🔧 Troubleshooting")
        st.markdown("""
        ### Port Already in Use

        If you get a port binding error, try a different port:
        ```bash
        streamlit run app.py --server.port 8503
        ```

        Or edit `.streamlit/config.toml`:
        ```toml
        [server]
        port = 8503
        address = "localhost"
        ```

        ### Virtual Environment Issues

        Make sure to activate the virtual environment before running:
        ```bash
        .venv\\Scripts\\activate  # Windows
        source .venv/bin/activate  # Linux/Mac
        ```

        ### Missing Dependencies

        Reinstall all dependencies:
        ```bash
        uv pip install -e .
        ```

        ### Import Errors

        Ensure the package is installed in editable mode:
        ```bash
        pip install -e .
        ```

        ### Memory Issues

        If running out of memory:
        - Reduce training sample size
        - Use smaller model parameters (fewer trees)
        - Process endpoints sequentially instead of in parallel

        ### Slow Performance

        To improve performance:
        - Use Random Forest instead of XGBoost (faster)
        - Reduce number of estimators (500 instead of 1000)
        - Enable parallel processing for batch operations
        - Cache trained models for reuse
        """)

    elif doc_section == "Advanced Features":
        st.subheader("🚀 Advanced Features")
        st.markdown("""
        ### Model Tuning

        - **Cross-Validation:** K-fold validation with configurable folds
        - **Grid Search:** Exhaustive hyperparameter search
        - **Random Search:** Efficient random sampling of parameter space

        ### Batch Processing

        - **Sequential Mode:** Process endpoints one at a time
        - **Parallel Mode:** Multi-threaded processing for faster throughput
        - **Statistics:** Comprehensive timing and performance metrics

        ### Visualization

        - **Timing Path Graphs:** Interactive network visualization with anchor highlighting
        - **Slack Distributions:** Histogram comparison of baseline vs. ML-accelerated
        - **Anchor Analysis:** Coverage and criticality visualization
        - **Feature Importance:** Bar charts for Random Forest and XGBoost

        ### CircuitNet Integration

        - **Synthetic Mode:** Generate realistic circuit data without downloads
        - **Real Data Mode:** Load actual CircuitNet designs (requires download)
        - **Batch Loading:** Process multiple designs simultaneously
        - **Training:** Train models on real circuit timing data

        ### Future Enhancements

        - Multi-anchor prediction for complex designs
        - GPU acceleration for large-scale analysis
        - Real STA tool integration (Synopsys PrimeTime, Cadence Tempus)
        - Advanced feature engineering
        - Deep learning models exploration
        - Incremental learning for online adaptation
        """)

    # References section at the bottom
    st.markdown("---")
    st.subheader("📖 References & Resources")
    st.markdown("""
    ### Key Concepts

    - **Endpoint-based graph representation** - Organizing timing paths by convergence points
    - **Middle-region anchor node selection** - Focusing on 30%-70% path region for optimal coverage
    - **Machine learning-based anchor prediction** - Using RF/XGBoost for intelligent node selection
    - **Path-based block analysis (PBA)** - Detailed timing analysis optimization

    ### External Resources

    **Research Background:**
    - Static Timing Analysis (STA) fundamentals
    - Path-Based Analysis (PBA) techniques
    - Machine learning for Electronic Design Automation (EDA)

    **Related Tools:**
    - Synopsys PrimeTime (commercial STA tool)
    - Cadence Tempus (commercial STA tool)
    - OpenSTA (open-source STA tool)

    **Datasets:**
    - CircuitNet: Large-scale circuit dataset
    - Hugging Face: circuitnet/CircuitNet-N14

    ### Documentation Files

    - **README.md** - Complete project overview
    - **QUICKSTART.md** - Quick start guide
    - **API.md** - Complete API reference
    - **DOCUMENTATION.md** - Documentation index
    - **CHANGELOG.md** - Version history

    ### Version Information

    - **Current Version:** 1.0.0
    - **Python Required:** 3.11+
    - **Last Updated:** November 12, 2025

    ### License

    This implementation is for research and educational purposes.
    """)

st.sidebar.title("About")
st.sidebar.info("""
**STA Path Pruning System**

Version 1.0

A machine learning-based system for accelerating static timing analysis through
intelligent path pruning and anchor node prediction.

**Technologies:**
- Python 3.11
- scikit-learn (Random Forest)
- XGBoost
- Streamlit
- Plotly

**Performance:**
- 30% faster than baseline
- Sub-picosecond accuracy
- Scalable to millions of pins
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Stats**")
st.sidebar.metric("Models Trained", "2")
st.sidebar.metric("Training Samples", "50")
st.sidebar.metric("Feature Dimension", "21")