"""
Step 2: Model Architecture Definition

This module defines the ML model architectures used for classifying mementos.
These models will be trained in Step 3 using user mementos from Firebase.

Key Components:
1. Category Classifier:
   - Multi-class classification model
   - Predicts memento category from text
   - Uses TF-IDF vectorization

2. Tag Predictor:
   - Multi-label classification model
   - Predicts relevant tags from text
   - Handles multiple tags per memento

3. Duration Estimator:
   - Multi-class classification model
   - Predicts time duration from text
   - Maps text to duration categories

Model Features:
- Text preprocessing and vectorization
- Hyperparameter configurations
- Evaluation metrics setup
- Model persistence utilities

Dependencies:
- scikit-learn: ML models and utilities
- numpy: Numerical operations
- pandas: Data handling
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class MementoModelTrainer:
    """
    Step 2: Model Training
    
    This class handles training and evaluation of machine learning models
    for memento categorization and tagging.
    """
    
    def __init__(self, 
                categories_path: str, 
                tags_path: str,
                durations_path: str,
                output_dir: str = "."):
        """
        Initialize the model trainer.
        
        Args:
            categories_path: Path to the JSON file containing category definitions
            tags_path: Path to the JSON file containing tag definitions
            durations_path: Path to the JSON file containing duration definitions
            output_dir: Directory to save the trained models
        """
        self.categories_path = categories_path
        self.tags_path = tags_path
        self.durations_path = durations_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load categories, tags, and durations
        self.categories, self.category_ids = self._load_categories()
        self.tags, self.tag_ids = self._load_tags()
        self.durations, self.duration_ids = self._load_durations()
        
        # Initialize models and vectorizer
        self.vectorizer = None
        self.category_model = None
        self.tags_model = None
        self.duration_model = None
    
    def _load_categories(self) -> Tuple[List[Dict], List[str]]:
        """Load categories from JSON file"""
        logging.info(f"Loading categories from {self.categories_path}")
        with open(self.categories_path, "r", encoding="utf-8") as f:
            categories = json.load(f)
            category_ids = [cat["symbol"] for cat in categories]
        logging.info(f"Loaded {len(categories)} categories")
        return categories, category_ids
    
    def _load_tags(self) -> Tuple[List[Dict], List[str]]:
        """Load tags from JSON file"""
        logging.info(f"Loading tags from {self.tags_path}")
        with open(self.tags_path, "r", encoding="utf-8") as f:
            tags = json.load(f)
            tag_ids = [tag["symbol"] for tag in tags]
        logging.info(f"Loaded {len(tags)} tags")
        return tags, tag_ids
    
    def _load_durations(self) -> Tuple[List[Dict], List[str]]:
        """Load durations from JSON file"""
        logging.info(f"Loading durations from {self.durations_path}")
        with open(self.durations_path, "r", encoding="utf-8") as f:
            durations = json.load(f)
            duration_ids = [duration["symbol"] for duration in durations]
        logging.info(f"Loaded {len(durations)} durations")
        return durations, duration_ids
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series]:
        """
        Load processed data for model training.
        
        Args:
            data_path: Path to the processed CSV file
            
        Returns:
            Tuple containing:
            - descriptions (DataFrame)
            - categories (Series)
            - tags (numpy array)
            - durations (Series)
        """
        logging.info(f"Loading processed data from {data_path}")
        try:
            df = pd.read_csv(data_path)
            
            # Combine text fields for ML
            df['text_for_ml'] = df.apply(
                lambda row: f"{row['name']} {row['description']} {row.get('caption', '')}", 
                axis=1
            )
            
            # Extract descriptions
            descriptions = df["text_for_ml"]
            
            # Extract categories
            categories = df["category"]
            
            # Extract durations
            durations = df["duration"]
            
            # Extract tags
            tags = df["tags"].apply(eval)  # Convert string representation of list to actual list
            
            # Convert tags to binary format
            mlb = MultiLabelBinarizer()
            tags_binary = mlb.fit_transform(tags)
            
            logging.info(f"Loaded {len(df)} samples with {tags_binary.shape[1]} possible tags")
            return descriptions, categories, tags_binary, durations
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def train_models(self, 
                     data_path: str, 
                     test_size: float = 0.2, 
                     random_state: int = 42,
                     use_grid_search: bool = False) -> Dict:
        """
        Train category, tag, and duration models.
        
        Args:
            data_path: Path to the processed data file
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Dictionary with training metrics
        """
        # Load data
        X, y_category, y_tags, y_duration = self.load_data(data_path)
        
        # Filter out empty durations for duration model training
        # Ensure duration values are strings and replace NaN with empty string
        y_duration = y_duration.fillna("").astype(str)
        duration_mask = y_duration.apply(lambda x: x.strip() != "")
        X_duration = X[duration_mask]
        y_duration_filtered = y_duration[duration_mask]
        
        if len(y_duration_filtered) > 0:
            logging.info(f"Filtered {len(y_duration_filtered)} samples with non-empty durations for duration model training")
        else:
            logging.warning("No samples with durations found, duration prediction will not be trained")
        
        # Split data for category and tag models
        X_train, X_test, y_cat_train, y_cat_test, y_tags_train, y_tags_test = train_test_split(
            X, y_category, y_tags, test_size=test_size, random_state=random_state
        )
        
        # Split data for duration model if we have duration data
        if len(y_duration_filtered) > 0:
            X_dur_train, X_dur_test, y_dur_train, y_dur_test = train_test_split(
                X_duration, y_duration_filtered, test_size=test_size, random_state=random_state
            )
            
            # Additional check to ensure no NaN values
            if y_dur_train.isna().any() or y_dur_test.isna().any():
                logging.warning("Detected NaN values in duration data. Cleaning...")
                y_dur_train = y_dur_train.fillna("")
                y_dur_test = y_dur_test.fillna("")
        
        logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        if len(y_duration_filtered) > 0:
            logging.info(f"Duration training set size: {len(X_dur_train)}, Duration test set size: {len(X_dur_test)}")
        
        # Create text vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Transform text data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train category model
        logging.info("Training category classification model...")
        self.category_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=random_state
        )
        self.category_model.fit(X_train_vec, y_cat_train)
        
        # Train tags model
        logging.info("Training tags classification model...")
        self.tags_model = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=random_state
            )
        )
        self.tags_model.fit(X_train_vec, y_tags_train)
        
        # Train duration model if we have duration data
        if len(y_duration_filtered) > 0:
            logging.info("Training duration prediction model...")
            X_dur_train_vec = self.vectorizer.transform(X_dur_train)
            X_dur_test_vec = self.vectorizer.transform(X_dur_test)
            
            self.duration_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=random_state
            )
            self.duration_model.fit(X_dur_train_vec, y_dur_train)
        
        # Save models
        self._save_models()
        
        # Evaluate models
        metrics = self._evaluate_models(X_test_vec, y_cat_test, y_tags_test)
        
        # Add duration metrics if available
        if len(y_duration_filtered) > 0:
            y_dur_pred = self.duration_model.predict(X_dur_test_vec)
            metrics["duration"] = {
                "accuracy": accuracy_score(y_dur_test, y_dur_pred),
                "report": classification_report(y_dur_test, y_dur_pred, output_dict=True)
            }
        
        return metrics
    
    def _evaluate_models(self, X_test_vec, y_cat_test, y_tags_test) -> Dict:
        """
        Evaluate trained models on test data.
        
        Args:
            X_test_vec: Vectorized test features
            y_cat_test: True category labels
            y_tags_test: True tag labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Evaluate category model
        y_cat_pred = self.category_model.predict(X_test_vec)
        cat_accuracy = accuracy_score(y_cat_test, y_cat_pred)
        cat_report = classification_report(y_cat_test, y_cat_pred, output_dict=True)
        
        # Log category metrics
        logging.info(f"Category Classification Accuracy: {cat_accuracy:.4f}")
        logging.info(f"Category Classification Report:\n{classification_report(y_cat_test, y_cat_pred)}")
        
        # Store category metrics
        metrics["category"] = {
            "accuracy": cat_accuracy,
            "report": cat_report
        }
        
        # Evaluate tags model
        y_tags_pred = self.tags_model.predict(X_test_vec)
        tags_f1_micro = f1_score(y_tags_test, y_tags_pred, average='micro')
        tags_f1_macro = f1_score(y_tags_test, y_tags_pred, average='macro')
        
        # Log tags metrics
        logging.info(f"Tags Classification F1 Score (micro): {tags_f1_micro:.4f}")
        logging.info(f"Tags Classification F1 Score (macro): {tags_f1_macro:.4f}")
        
        # Store tags metrics
        metrics["tags"] = {
            "f1_micro": tags_f1_micro,
            "f1_macro": tags_f1_macro
        }
        
        return metrics
    
    def _save_models(self):
        """Save trained models and vectorizer to disk"""
        # Create paths
        vectorizer_path = os.path.join(self.output_dir, "vectorizer.pkl")
        category_model_path = os.path.join(self.output_dir, "category_model.pkl")
        tags_model_path = os.path.join(self.output_dir, "tags_model.pkl")
        duration_model_path = os.path.join(self.output_dir, "duration_model.pkl")
        
        # Save vectorizer
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logging.info(f"Saved vectorizer to {vectorizer_path}")
        
        # Save category model
        with open(category_model_path, "wb") as f:
            pickle.dump(self.category_model, f)
        logging.info(f"Saved category model to {category_model_path}")
        
        # Save tags model
        with open(tags_model_path, "wb") as f:
            pickle.dump(self.tags_model, f)
        logging.info(f"Saved tags model to {tags_model_path}")
        
        # Save duration model if available
        if self.duration_model is not None:
            with open(duration_model_path, "wb") as f:
                pickle.dump(self.duration_model, f)
            logging.info(f"Saved duration model to {duration_model_path}")
    
    def save_model_info(self, metrics: Dict):
        """Save model information and metrics to JSON file"""
        # Create model info
        model_info = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "vectorizer": {
                "max_features": self.vectorizer.max_features,
                "ngram_range": str(self.vectorizer.ngram_range),
                "vocabulary_size": len(self.vectorizer.vocabulary_)
            },
            "category_model": {
                "type": type(self.category_model).__name__,
                "params": str(self.category_model.get_params())
            },
            "tags_model": {
                "type": type(self.tags_model).__name__,
                "params": str(self.tags_model.get_params())
            },
            "metrics": self._convert_metrics_for_json(metrics)
        }
        
        # Add duration model info if available
        if self.duration_model is not None:
            model_info["duration_model"] = {
                "type": type(self.duration_model).__name__,
                "params": str(self.duration_model.get_params())
            }
        
        # Save model info
        model_info_path = os.path.join(self.output_dir, "model_info.json")
        with open(model_info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2)
        logging.info(f"Saved model info to {model_info_path}")
    
    def _convert_metrics_for_json(self, metrics: Dict) -> Dict:
        """Convert metrics dictionary to be JSON serializable"""
        json_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                json_metrics[key] = self._convert_metrics_for_json(value)
            elif hasattr(value, 'tolist'):
                # Convert numpy arrays to lists
                json_metrics[key] = value.tolist()
            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                # Convert other non-serializable objects to strings
                json_metrics[key] = str(value)
            else:
                json_metrics[key] = value
                
        return json_metrics

def main():
    """Example usage of the MementoModelTrainer class"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Set up paths
    categories_path = os.path.join(root_dir, "memento_categories_combined.json")
    tags_path = os.path.join(root_dir, "memento_tags_combined.json")
    durations_path = os.path.join(root_dir, "memento_durations.json")
    data_dir = os.path.join("ml_pipeline", "output", "step1_data_processing", "processed_data")
    models_dir = os.path.join("ml_pipeline", "output", "step2_model_training", "models")
    metrics_dir = os.path.join("ml_pipeline", "output", "step2_model_training", "metrics")
    
    # Create output directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logging.error(f"Data directory {data_dir} not found or empty. Please run step1_data_loader.py first.")
        return
    
    # Check if processed data file exists
    processed_data_path = os.path.join(data_dir, "user_mementos_processed.csv")
    if not os.path.exists(processed_data_path):
        logging.error(f"Processed data file {processed_data_path} not found. Please run step1_data_loader.py first.")
        return
    
    # Initialize model trainer
    trainer = MementoModelTrainer(
        categories_path=categories_path,
        tags_path=tags_path,
        durations_path=durations_path,
        output_dir=models_dir
    )
    
    # Train models
    try:
        metrics = trainer.train_models(
            data_path=processed_data_path,
            test_size=0.2,
            random_state=42,
            use_grid_search=False  # Set to True for hyperparameter tuning
        )
        
        # Save model info
        trainer.save_model_info(metrics)
        
        # Save metrics
        metrics_path = os.path.join(metrics_dir, "model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logging.info("Model training completed successfully!")
    except Exception as e:
        logging.error(f"Error training models: {e}")

if __name__ == "__main__":
    main() 