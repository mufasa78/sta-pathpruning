import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from typing import List, Tuple, Dict, Any
from .data_structures import EndpointBasedGraph, Node
from .anchor_predictor import AnchorPredictor
from .feature_extractor import FeatureExtractor


class ModelTuner:
    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        self.best_params = None
        self.cv_results = None
        self.feature_extractor = FeatureExtractor()
    
    def get_default_param_grid(self) -> Dict[str, List[Any]]:
        if self.model_type == 'rf':
            return {
                'n_estimators': [100, 500, 1000],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        elif self.model_type == 'xgb':
            return {
                'n_estimators': [100, 500, 1000],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_training_arrays(self, 
                                training_data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
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
        
        return X, y
    
    def grid_search_cv(self, 
                       training_data: List[Tuple],
                       param_grid: Dict[str, List[Any]] = None,
                       cv: int = 5,
                       n_jobs: int = -1) -> Dict[str, Any]:
        X, y = self.prepare_training_arrays(training_data)
        
        if param_grid is None:
            param_grid = self.get_default_param_grid()
        
        if self.model_type == 'rf':
            base_model = RandomForestClassifier(random_state=42, n_jobs=1)
        else:
            base_model = xgb.XGBClassifier(random_state=42, n_jobs=1)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.cv_results = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }
        
        return self.cv_results
    
    def random_search_cv(self,
                        training_data: List[Tuple],
                        param_distributions: Dict[str, List[Any]] = None,
                        n_iter: int = 20,
                        cv: int = 5,
                        n_jobs: int = -1) -> Dict[str, Any]:
        X, y = self.prepare_training_arrays(training_data)
        
        if param_distributions is None:
            param_distributions = self.get_default_param_grid()
        
        if self.model_type == 'rf':
            base_model = RandomForestClassifier(random_state=42, n_jobs=1)
        else:
            base_model = xgb.XGBClassifier(random_state=42, n_jobs=1)
        
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='f1',
            n_jobs=n_jobs,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        self.cv_results = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'cv_results': random_search.cv_results_
        }
        
        return self.cv_results
    
    def cross_validate(self,
                      training_data: List[Tuple],
                      params: Dict[str, Any] = None,
                      cv: int = 5) -> Dict[str, Any]:
        X, y = self.prepare_training_arrays(training_data)
        
        if params is None:
            params = {'n_estimators': 500, 'max_depth': 20, 'random_state': 42}
        
        if self.model_type == 'rf':
            model = RandomForestClassifier(**params, n_jobs=-1)
        else:
            model = xgb.XGBClassifier(**params, n_jobs=-1)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores.tolist(),
            'params': params
        }
    
    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params if self.best_params else {}
    
    def create_tuned_predictor(self) -> AnchorPredictor:
        if self.best_params is None:
            raise ValueError("No hyperparameter tuning has been performed yet")
        
        predictor = AnchorPredictor(model_type=self.model_type)
        predictor.model = predictor._create_model()
        
        for param, value in self.best_params.items():
            if hasattr(predictor.model, param):
                setattr(predictor.model, param, value)
        
        return predictor
