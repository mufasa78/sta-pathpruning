import numpy as np
from typing import List, Optional
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from .data_structures import Node, EndpointBasedGraph
from .feature_extractor import FeatureExtractor


class AnchorPredictor:
    def __init__(self, model_type: str = 'rf', n_estimators: int = 1000):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self._is_trained = False
    
    def _create_model(self):
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgb':
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, training_data: List[tuple]):
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        X_list = []
        y_list = []
        
        for endpoint_graph, candidates, true_anchor_idx in training_data:
            features = self.feature_extractor.extract_batch(candidates, endpoint_graph)
            
            labels = np.zeros(len(candidates), dtype=int)
            if 0 <= true_anchor_idx < len(candidates):
                labels[true_anchor_idx] = 1
            
            X_list.append(features)
            y_list.append(labels)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        self.model = self._create_model()
        self.model.fit(X, y)
        self._is_trained = True
        
        return self
    
    def predict(self, candidates: List[Node], endpoint_graph: EndpointBasedGraph) -> Optional[Node]:
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        if not self._is_trained:
            return self._fallback_prediction(candidates, endpoint_graph)
        
        features = self.feature_extractor.extract_batch(candidates, endpoint_graph)
        
        if self.model_type == 'rf':
            probs = self.model.predict_proba(features)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(features)
        else:
            probs = self.model.predict_proba(features)[:, 1]
        
        best_idx = np.argmax(probs)
        return candidates[best_idx]
    
    def _fallback_prediction(self, candidates: List[Node], endpoint_graph: EndpointBasedGraph) -> Node:
        scored_candidates = []
        for node in candidates:
            paths_through = endpoint_graph.get_paths_through_node(node)
            coverage = len(paths_through) / max(endpoint_graph.num_paths, 1)
            avg_slack = np.mean([p.slack for p in paths_through]) if paths_through else 0
            score = coverage * 0.7 + (-avg_slack) * 0.3
            scored_candidates.append((node, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]
    
    def get_feature_importance(self) -> Optional[dict]:
        if not self._is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        feature_names = self.feature_extractor.feature_names
        
        return dict(zip(feature_names, importances))
