"""
Step 3: Model Training

This module trains the ML models defined in Step 2 using user mementos from Step 1.
The trained models will be used to classify scraped data in Step 5.

Key Functions:
1. Data Loading:
   - Loads processed user mementos from Step 1
   - Prepares features and labels for training

2. Model Training:
   - Trains category classifier
   - Trains tag predictor
   - Trains duration estimator
   - Uses cross-validation for evaluation

3. Model Evaluation:
   - Computes accuracy metrics
   - Generates classification reports
   - Creates confusion matrices
   - Validates model performance

4. Model Persistence:
   - Saves trained models to output/step3_model_training/
   - Saves evaluation metrics and reports
   - Handles model versioning

Output Files:
- output/step3_model_training/
  - category_model.pkl: Trained category classifier
  - tags_model.pkl: Trained tag predictor
  - duration_model.pkl: Trained duration estimator
  - vectorizer.pkl: Fitted text vectorizer
  - model_metrics.json: Evaluation metrics

Dependencies:
- scikit-learn: ML training and evaluation
- pandas: Data handling
- numpy: Numerical operations
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class MementoModelTrainer:
    """
    Step 3: Train ML Models
    
    This class handles training machine learning models using user mementos
    from Firebase to classify categories, tags, and durations.
    """
    
    def __init__(self, 
                 input_dir: str = "ml_pipeline/output/step1_data_processing/processed_data",
                 output_dir: str = "ml_pipeline/output/step3_model_training"):
        """Initialize the trainer"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, "models")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.reports_dir = os.path.join(output_dir, "reports")
        
        self.vectorizer = None
        self.category_model = None
        self.tags_model = None
        self.duration_model = None
        
        # Create output directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Define hyperparameter grids
        self.param_grid = {
            'C': [0.1, 1, 10],
            'max_iter': [1000],
            'class_weight': ['balanced'],
            'solver': ['liblinear', 'saga']
        }
    
    def _tune_hyperparameters(self, model, X: np.ndarray, y: np.ndarray, is_multilabel: bool = False) -> Dict:
        """
        Tune hyperparameters using grid search or simple validation split for small datasets.
        
        Args:
            model: Base model to tune
            X: Training features
            y: Target variable
            is_multilabel: Whether this is a multi-label classification problem
            
        Returns:
            Dictionary with best parameters
        """
        logging.info("Tuning hyperparameters...")
        
        # For multi-label classification or very small datasets, use default parameters
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        if is_multilabel or n_samples < 20:
            logging.warning("Using default parameters due to multi-label classification or small dataset")
            best_params = {
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'solver': 'liblinear'
            }
            if hasattr(model, 'set_params'):
                model.set_params(**best_params)
            model.fit(X, y)
            y_pred = model.predict(X)
            if is_multilabel:
                best_score = f1_score(y, y_pred, average='samples')
            else:
                best_score = f1_score(y, y_pred, average='weighted')
        else:
            # For single-label classification with enough samples
            unique_labels = np.unique(y)
            min_samples = min([sum(y == label) for label in unique_labels])
            
            if min_samples >= 2:
                try:
                    grid_search = GridSearchCV(
                        model,
                        self.param_grid,
                        cv=2,
                        scoring='f1_weighted',
                        n_jobs=-1
                    )
                    grid_search.fit(X, y)
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                except Exception as e:
                    logging.warning(f"Grid search failed, using default parameters: {e}")
                    best_params = {
                        'C': 1.0,
                        'class_weight': 'balanced',
                        'max_iter': 1000,
                        'solver': 'liblinear'
                    }
                    if hasattr(model, 'set_params'):
                        model.set_params(**best_params)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    best_score = f1_score(y, y_pred, average='weighted')
            else:
                logging.warning("Using default parameters due to insufficient samples per class")
                best_params = {
                    'C': 1.0,
                    'class_weight': 'balanced',
                    'max_iter': 1000,
                    'solver': 'liblinear'
                }
                if hasattr(model, 'set_params'):
                    model.set_params(**best_params)
                model.fit(X, y)
                y_pred = model.predict(X)
                best_score = f1_score(y, y_pred, average='weighted')
        
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score:.3f}")
        
        return best_params
    
    def _validate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, is_multilabel: bool = False) -> Dict:
        """
        Validate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            is_multilabel: Whether this is a multi-label classification problem
            
        Returns:
            Dictionary with validation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if is_multilabel:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        else:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
        
        # Skip cross-validation for multi-label or very small datasets
        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
        }
        
        if not is_multilabel:
            metrics["confusion_matrix"] = cm.tolist()
        
        # Skip cross-validation for multi-label or small datasets
        n_samples = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
        if not is_multilabel and n_samples >= 20:
            unique_labels = np.unique(y_test)
            min_samples = min([sum(y_test == label) for label in unique_labels])
            
            if min_samples >= 2:
                try:
                    cv_scores = cross_val_score(model, X_test, y_test, cv=2)
                    metrics.update({
                        "cv_scores": cv_scores.tolist(),
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std()
                    })
                except Exception as e:
                    logging.warning(f"Skipping cross-validation due to insufficient data: {e}")
                    metrics.update({
                        "cv_scores": [],
                        "cv_mean": accuracy,
                        "cv_std": 0.0
                    })
            else:
                logging.warning("Skipping cross-validation due to insufficient samples per class")
                metrics.update({
                    "cv_scores": [],
                    "cv_mean": accuracy,
                    "cv_std": 0.0
                })
        else:
            logging.warning("Skipping cross-validation for multi-label classification or small dataset")
            metrics.update({
                "cv_scores": [],
                "cv_mean": accuracy,
                "cv_std": 0.0
            })
        
        return metrics
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load processed user mementos for training.
        
        Returns:
            DataFrame with training data
        """
        input_file = os.path.join(self.input_dir, "user_mementos_processed.csv")
        logging.info(f"Loading training data from {input_file}")
        
        try:
            df = pd.read_csv(input_file)
            logging.info(f"Loaded {len(df)} training examples")
            return df
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            return pd.DataFrame()
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with training data
            
        Returns:
            Tuple of (text features, target variables)
        """
        logging.info("Preparing training data...")
        
        # Combine text fields for ML
        df['text_for_ml'] = df.apply(
            lambda row: f"{row['name']} {row['description']} {row.get('caption', '')}", 
            axis=1
        )
        
        # Extract text features
        texts = df["text_for_ml"].fillna("")
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        
        # Transform texts
        X = self.vectorizer.fit_transform(texts)
        logging.info(f"Vectorized {X.shape[0]} texts with {X.shape[1]} features")
        
        # Convert tags to binary matrix
        mlb = MultiLabelBinarizer()
        tags_matrix = mlb.fit_transform([eval(tags) if isinstance(tags, str) else tags for tags in df["tags"]])
        
        # Extract target variables
        y = {
            "category": df["category"].values,
            "tags": tags_matrix,
            "duration": df["duration"].values
        }
        
        return X, y
    
    def train_category_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train category classification model.
        
        Args:
            X: Text features
            y: Category labels
            
        Returns:
            Dictionary with model performance metrics
        """
        logging.info("Training category model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize base model
        base_model = LogisticRegression()
        
        # Tune hyperparameters
        best_params = self._tune_hyperparameters(base_model, X_train, y_train)
        
        # Train final model
        model = LogisticRegression(**best_params)
        model.fit(X_train, y_train)
        
        # Validate model
        metrics = self._validate_model(model, X_test, y_test)
        
        # Save model
        self.category_model = model
        model_path = os.path.join(self.models_dir, "01_category_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logging.info(f"Category model saved to {model_path}")
        
        return metrics
    
    def train_tags_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train tags classification model"""
        logging.info("Training tags model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # For multi-label classification, use default parameters
        base_params = {
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'solver': 'liblinear'
        }
        
        # Check if we have enough classes for each tag
        valid_tags = []
        for i in range(y_train.shape[1]):
            unique_classes = np.unique(y_train[:, i])
            if len(unique_classes) >= 2:
                valid_tags.append(i)
        
        if not valid_tags:
            logging.warning("No tags have enough classes for training. Skipping tags model.")
        return {
                "accuracy": 0.0,
                "classification_report": {},
                "cv_scores": [],
                "cv_mean": 0.0,
                "cv_std": 0.0
            }
        
        # Select only valid tags
        y_train = y_train[:, valid_tags]
        y_test = y_test[:, valid_tags]
        
        # Initialize and train model
        model = MultiOutputClassifier(LogisticRegression(**base_params))
        model.fit(X_train, y_train)
        
        # Validate model
        metrics = self._validate_model(model, X_test, y_test, is_multilabel=True)
        
        # Save model
        model_path = os.path.join(self.models_dir, "02_tags_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                'model': model,
                'valid_tags': valid_tags
            }, f)
        logging.info(f"Tags model saved to {model_path}")
        
        return metrics
    
    def train_duration_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train duration classification model"""
        logging.info("Training duration model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize base model
        base_model = LogisticRegression()
        
        # Tune hyperparameters
        best_params = self._tune_hyperparameters(base_model, X_train, y_train)
        
        # Train final model
        model = LogisticRegression(**best_params)
        model.fit(X_train, y_train)
        
        # Validate model
        metrics = self._validate_model(model, X_test, y_test)
        
        # Save model
        model_path = os.path.join(self.models_dir, "02_duration_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Duration model saved to {model_path}")
        
        return metrics
    
    def save_vectorizer(self):
        """Save the text vectorizer"""
        if self.vectorizer is None:
            logging.error("Vectorizer not initialized")
            return
        
        vectorizer_path = os.path.join(self.models_dir, "01_vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        logging.info(f"Vectorizer saved to {vectorizer_path}")
    
    def generate_model_report(self, metrics: Dict) -> None:
        """Generate a report of model performance"""
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "category": {
                    "accuracy": metrics["category"]["accuracy"],
                    "f1_weighted": metrics["category"]["classification_report"]["weighted avg"]["f1-score"],
                    "cv_mean": metrics["category"]["cv_mean"],
                    "cv_std": metrics["category"]["cv_std"]
                },
                "tags": {
                    "accuracy": metrics["tags"]["accuracy"],
                    "f1_weighted": metrics["tags"]["classification_report"]["weighted avg"]["f1-score"] if "weighted avg" in metrics["tags"]["classification_report"] else 0.0,
                    "cv_mean": metrics["tags"]["cv_mean"],
                    "cv_std": metrics["tags"]["cv_std"]
                },
                "duration": {
                    "accuracy": metrics["duration"]["accuracy"],
                    "f1_weighted": metrics["duration"]["classification_report"]["weighted avg"]["f1-score"],
                    "cv_mean": metrics["duration"]["cv_mean"],
                    "cv_std": metrics["duration"]["cv_std"]
                }
            }
        }
        
        # Save report
        report_path = os.path.join(self.reports_dir, "model_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logging.info(f"Model report saved to {report_path}")
    
    def train_models(self):
        """Train all models"""
        # Load training data
        df = self.load_training_data()
        if df.empty:
            logging.error("No training data available")
            return
        
        # Prepare training data
        X, y = self.prepare_training_data(df)
        
        # Train models
        metrics = {
            "category": self.train_category_model(X, y["category"]),
            "tags": self.train_tags_model(X, y["tags"]),
            "duration": self.train_duration_model(X, y["duration"])
        }
        
        # Save vectorizer
        self.save_vectorizer()
        
        # Generate model report
        self.generate_model_report(metrics)
        
        logging.info("Model training complete")

def main():
    """Main function"""
    # Initialize trainer
    trainer = MementoModelTrainer(
        input_dir="ml_pipeline/output/step1_data_processing/processed_data",
        output_dir="ml_pipeline/output/step3_model_training"
    )
    
    # Train models
    trainer.train_models()

if __name__ == "__main__":
    main() 