"""
Step 5: Process Scraped Data

This module processes scraped data from Step 4 through trained models from Step 3.
It classifies each scraped memento into categories, tags, and durations.

Key Functions:
1. Data Loading:
   - Loads scraped data from Step 4
   - Loads trained models from Step 3
   - Prepares data for processing

2. Data Processing:
   - Preprocesses text data
   - Applies trained models:
     - Category classification
     - Tag prediction
     - Duration estimation
   - Validates predictions

3. Quality Control:
   - Checks prediction confidence
   - Validates classification results
   - Filters low-confidence predictions
   - Generates processing reports

4. Data Storage:
   - Saves processed data to output/step5_processed_data/
   - Includes original data and predictions
   - Stores confidence scores
   - Maintains processing logs

Output Files:
- output/step5_processed_data/
  - classified_mementos.csv: Processed data with predictions
  - confidence_scores.json: Prediction confidence metrics
  - processing_report.json: Processing statistics
  - validation_log.txt: Validation details

Dependencies:
- pandas: Data processing
- scikit-learn: Model loading
- numpy: Numerical operations
"""

import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class ScrapedDataProcessor:
    """
    Step 4: Process Scraped Data
    
    This class handles processing scraped data through trained ML models
    to classify them into categories, tags, and durations.
    """
    
    def __init__(self, 
                 model_dir: str = "ml_pipeline/output/step3_model_training/models",
                 output_dir: str = "ml_pipeline/output/step5_processed_data"):
        """Initialize the processor"""
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.processed_data_dir = os.path.join(output_dir, "processed_data")
        self.reports_dir = os.path.join(output_dir, "reports")
        self.validation_dir = os.path.join(output_dir, "validation")
        
        self.vectorizer = None
        self.category_model = None
        self.tags_model = None
        self.duration_model = None
        
        # Confidence thresholds
        self.confidence_thresholds = {
            "category": 0.4,
            "tags": 0.3,
            "duration": 0.4
        }
        
        # Create output directories
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load trained models"""
        try:
            # Load vectorizer
            with open(os.path.join(self.model_dir, "01_vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            logging.info("Loaded vectorizer")
            
            # Load category model
            with open(os.path.join(self.model_dir, "01_category_model.pkl"), "rb") as f:
                self.category_model = pickle.load(f)
            logging.info("Loaded category model")
            
            # Load tags model
            with open(os.path.join(self.model_dir, "02_tags_model.pkl"), "rb") as f:
                tags_data = pickle.load(f)
                self.tags_model = tags_data['model']
                self.valid_tags = tags_data['valid_tags']
                
                # Load training data to get tag names
                training_file = os.path.join("ml_pipeline/output/step1_data_processing/processed_data", "user_mementos_processed.csv")
                if os.path.exists(training_file):
                    df = pd.read_csv(training_file)
                    # Get all unique tags from training data
                    all_tags = set()
                    for tags in df["tags"]:
                        if isinstance(tags, str):
                            tags = eval(tags)
                        all_tags.update(tags)
                    self.tag_names = sorted(list(all_tags))
                    logging.info(f"Loaded {len(self.tag_names)} tag names from training data")
                else:
                    logging.warning("Training data not found, using tag indices")
                    self.tag_names = [f"tag_{i}" for i in range(len(self.valid_tags))]
            logging.info("Loaded tags model")
            
            # Load duration model
            with open(os.path.join(self.model_dir, "02_duration_model.pkl"), "rb") as f:
                self.duration_model = pickle.load(f)
            logging.info("Loaded duration model")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def load_scraped_data(self, json_path: str) -> pd.DataFrame:
        """
        Load scraped data from JSON file.
        
        Args:
            json_path: Path to the JSON file containing scraped data
            
        Returns:
            DataFrame with scraped data
        """
        logging.info(f"Loading scraped data from {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                scraped_data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(scraped_data)
            logging.info(f"Loaded {len(df)} scraped mementos")
            return df
            
        except Exception as e:
            logging.error(f"Error loading scraped data: {e}")
            return pd.DataFrame()
    
    def _extract_text_for_ml(self, memento: Dict) -> str:
        """
        Extract text for ML from memento.
        
        Args:
            memento: Dictionary containing memento data
            
        Returns:
            String containing text for ML processing
        """
        text_parts = []
        
        # Add name with higher weight (repeat 5 times for more emphasis)
        name = memento.get('name', '')
        text_parts.extend([name] * 5)
        
        # Add description with medium weight (repeat 2 times)
        description = memento.get('description', '')
        text_parts.extend([description] * 2)
        
        # Add location name if available (repeat 2 times)
        location = memento.get('location', {})
        if isinstance(location, dict):
            location_str = str(location)
            text_parts.extend([location_str] * 2)
        
        # Add tags if available (repeat 3 times)
        tags = memento.get('tags', [])
        if tags:
            tags_str = ' '.join(tags)
            text_parts.extend([tags_str] * 3)
        
        # Add category if available (repeat 3 times)
        category = memento.get('category', '')
        if category:
            text_parts.extend([category] * 3)
        
        # Join all parts with space and clean
        text = ' '.join(filter(None, text_parts))
        
        # Additional text cleaning
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.strip()
        
        return text
    
    def _validate_prediction(self, 
                           prediction: str,
                           confidence: float,
                           threshold: float) -> Tuple[bool, str]:
        """
        Validate a prediction based on confidence threshold.
        
        Args:
            prediction: Model prediction
            confidence: Prediction confidence
            threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if confidence < threshold:
            return False, f"Low confidence: {confidence:.2f} < {threshold}"
        return True, "Valid prediction"
    
    def _validate_predictions(self, 
                            category_pred: str,
                            category_conf: float,
                            tags_pred: List[str],
                            tags_conf: float,
                            duration_pred: str,
                            duration_conf: float) -> Dict:
        """
        Validate all predictions for a memento.
        
        Args:
            category_pred: Predicted category
            category_conf: Category confidence
            tags_pred: Predicted tags
            tags_conf: Tags confidence
            duration_pred: Predicted duration
            duration_conf: Duration confidence
            
        Returns:
            Dictionary with validation results
        """
        # Validate category
        cat_valid, cat_reason = self._validate_prediction(
            category_pred,
            category_conf,
            self.confidence_thresholds["category"]
        )
        
        # Validate tags
        tags_valid, tags_reason = self._validate_prediction(
            str(tags_pred),
            tags_conf,
            self.confidence_thresholds["tags"]
        )
        
        # Validate duration
        dur_valid, dur_reason = self._validate_prediction(
            duration_pred,
            duration_conf,
            self.confidence_thresholds["duration"]
        )
        
        # Overall validation
        is_valid = all([cat_valid, tags_valid, dur_valid])
        reasons = [cat_reason, tags_reason, dur_reason]
        
        return {
            "is_valid": is_valid,
            "validation_details": {
                "category": {"valid": cat_valid, "reason": cat_reason},
                "tags": {"valid": tags_valid, "reason": tags_reason},
                "duration": {"valid": dur_valid, "reason": dur_reason}
            },
            "reasons": reasons
        }
    
    def process_memento(self, memento: Dict) -> Dict:
        """
        Process a single memento through ML models.
        
        Args:
            memento: Dictionary containing memento data
            
        Returns:
            Dictionary with original data and predictions
        """
        try:
            # Extract text for ML
            text = self._extract_text_for_ml(memento)
            
            # Vectorize text
            text_vector = self.vectorizer.transform([text])
            
            # Get category prediction with probability
            category_probs = self.category_model.predict_proba(text_vector)[0]
            category_pred = self.category_model.predict(text_vector)[0]
            category_conf = float(max(category_probs))
            
            # Get tags prediction with probabilities
            tags_pred_matrix = self.tags_model.predict(text_vector)
            tags_conf_matrix = self.tags_model.predict_proba(text_vector)
            
            # Convert tag predictions to list with confidence scores
            tags_pred = []
            tags_conf = []
            for i, pred in enumerate(tags_pred_matrix[0]):
                if pred == 1:
                    tag_idx = self.valid_tags[i]
                    if tag_idx < len(self.tag_names):
                        tags_pred.append(self.tag_names[tag_idx])
                        tags_conf.append(float(max(tags_conf_matrix[i][0])))
            
            # Get duration prediction with probability
            duration_probs = self.duration_model.predict_proba(text_vector)[0]
            duration_pred = self.duration_model.predict(text_vector)[0]
            duration_conf = float(max(duration_probs))
            
            # Validate predictions
            validation = self._validate_predictions(
                category_pred,
                category_conf,
                tags_pred,
                np.mean(tags_conf) if tags_conf else 0.0,
                duration_pred,
                duration_conf
            )
            
            # Create processed memento with all original data
            processed_memento = {
                # Original memento data
                'userId': memento.get('userId', 'public'),
                'location': memento.get('location', {}),
                'media': memento.get('media', []),
                'name': memento.get('name', ''),
                'description': memento.get('description', ''),
                'category': memento.get('category', ''),
                'timestamp': memento.get('timestamp', ''),
                'tags': memento.get('tags', []),
                'link': memento.get('link', ''),
                'mementoType': memento.get('mementoType', 'public'),
                'duration': memento.get('duration', ''),
                
                # ML predictions with confidence scores
                'predicted_category': category_pred,
                'category_confidence': category_conf,
                'predicted_tags': tags_pred,
                'tags_confidence': float(np.mean(tags_conf) if tags_conf else 0.0),
                'predicted_duration': duration_pred,
                'duration_confidence': duration_conf,
                'validation_results': validation,
                'processing_status': 'success'
            }
            
            return processed_memento
            
        except Exception as e:
            logging.error(f"Error processing memento: {e}")
            # Return original memento with error status
            memento['processing_status'] = 'error'
            memento['error_message'] = str(e)
            return memento
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all scraped data through ML models.
        
        Args:
            df: DataFrame containing scraped data
            
        Returns:
            DataFrame with processed data and predictions
        """
        logging.info("Processing scraped data through ML models...")
        
        # Process each memento
        processed_data = []
        for _, memento in df.iterrows():
            processed = self.process_memento(memento.to_dict())
            processed_data.append(processed)
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        # Calculate processing statistics
        total = len(processed_df)
        successful = len(processed_df[processed_df["processing_status"] == "success"])
        failed = len(processed_df[processed_df["processing_status"] == "error"])
        
        logging.info(f"Processing complete:")
        logging.info(f"- Total mementos: {total}")
        logging.info(f"- Successfully processed: {successful}")
        logging.info(f"- Failed to process: {failed}")
        
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_scraped_data.csv"):
        """
        Save processed data to CSV file.
        
        Args:
            df: DataFrame with processed data
            filename: Output filename
        """
        try:
            # Ensure all required columns are present
            required_columns = [
                'userId', 'location', 'media', 'name', 'description', 
                'category', 'timestamp', 'mementoTags', 'link', 'mementoType',
                'predicted_category', 'category_confidence',
                'predicted_tags', 'tags_confidence',
                'predicted_duration', 'duration_confidence',
                'validation_results'
            ]
            
            # Add any missing columns with default values
            for col in required_columns:
                if col not in df.columns:
                    if col in ['userId', 'mementoType']:
                        df[col] = 'public'
                    elif col in ['location', 'media']:
                        df[col] = '{}' if col == 'location' else '[]'  # Use proper JSON format
                    else:
                        df[col] = ''
            
            # Save to CSV
            output_path = os.path.join(self.processed_data_dir, filename)
            df.to_csv(output_path, index=False)
            logging.info(f"Saved processed data to {output_path}")
            
            # Also save as JSON for easier integration
            json_path = os.path.join(self.processed_data_dir, "processed_scraped_data.json")
            df.to_json(json_path, orient='records', indent=2)
            logging.info(f"Saved processed data to {json_path}")
            
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")
            raise
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate processing report.
        
        Args:
            df: DataFrame containing processed data
            
        Returns:
            Dictionary containing processing statistics
        """
        # Calculate basic statistics
        total = len(df)
        successful = len(df[df["processing_status"] == "success"])
        failed = len(df[df["processing_status"] == "error"])
        
        # Calculate category distribution
        category_dist = df[df["processing_status"] == "success"]["predicted_category"].value_counts().to_dict()
        
        # Calculate duration distribution
        duration_dist = df[df["processing_status"] == "success"]["predicted_duration"].value_counts().to_dict()
        
        # Calculate average confidence scores
        avg_category_conf = df[df["processing_status"] == "success"]["category_confidence"].mean()
        avg_tags_conf = df[df["processing_status"] == "success"]["tags_confidence"].mean()
        avg_duration_conf = df[df["processing_status"] == "success"]["duration_confidence"].mean()
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_mementos": total,
            "successful_processing": successful,
            "failed_processing": failed,
            "success_rate": successful / total if total > 0 else 0,
            "category_distribution": category_dist,
            "duration_distribution": duration_dist,
            "average_confidence": {
                "category": float(avg_category_conf),
                "tags": float(avg_tags_conf),
                "duration": float(avg_duration_conf)
            }
        }
        
        # Save report
        report_path = os.path.join(self.reports_dir, "processing_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Generated processing report: {report_path}")
        return report

def main():
    """Main function"""
    # Initialize processor
    processor = ScrapedDataProcessor()
    
    # Load scraped data
    scraped_data_path = os.path.join("ml_pipeline", "output", "step4_scraped_data", "raw_data", "scraped_mementos.json")
    scraped_df = processor.load_scraped_data(scraped_data_path)
    
    if scraped_df.empty:
        logging.error("No scraped data found")
        return
    
    # Process data
    processed_df = processor.process_data(scraped_df)
    
    # Save processed data
    processor.save_processed_data(processed_df)
    
    # Generate report
    processor.generate_report(processed_df)

if __name__ == "__main__":
    main() 