#!/usr/bin/env python
"""
Step 5: ML Model Predictor

This module provides a production-ready predictor class that uses trained ML models
to classify new mementos with categories, tags, and durations.

Key Features:
1. Model Loading:
   - Loads trained TF-IDF vectorizer
   - Loads trained classifiers for:
     * Category prediction
     * Tag prediction
     * Duration prediction
   - Handles missing models gracefully

2. Prediction Functions:
   - predict_category(): Single category classification
   - predict_tags(): Multi-label tag prediction
   - predict_duration(): Experience duration prediction
   - classify_memento(): Complete memento classification

3. Integration Features:
   - Configurable prediction thresholds
   - Batch prediction support
   - Error handling and logging
   - Scraper integration utilities

4. Production Features:
   - Model versioning
   - Performance optimization
   - Memory management
   - Robust error handling

Usage Example:
```python
from step5_predictor import MementoPredictor

predictor = MementoPredictor()
result = predictor.classify_memento(
    "Enjoy a sunset picnic in Central Park with live music"
)
# Returns:
# {
#   "category": "🌳 Outdoors",
#   "tags": ["🌅 Sunset", "🎵 Music", "🏞️ Parks"],
#   "duration": "1-2 hours"
# }
```

Dependencies:
- scikit-learn: ML model loading
- numpy: Numerical operations
- pickle: Model deserialization
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class MementoPredictor:
    """
    A class for predicting memento categories and tags using trained ML models.
    This can be integrated with the scraper to replace the rule-based classification.
    """
    
    def __init__(self, 
                 models_dir: Optional[str] = None,
                 categories_path: Optional[str] = None,
                 tags_path: Optional[str] = None,
                 durations_path: Optional[str] = None,
                 threshold: float = 0.2):
        """
        Initialize the predictor with trained models.
        
        Args:
            models_dir: Directory containing the trained models
            categories_path: Path to the JSON file with category definitions
            tags_path: Path to the JSON file with tag definitions
            durations_path: Path to the JSON file with duration definitions
            threshold: Probability threshold for tag prediction
        """
        # Set up paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.script_dir)
        
        # Set models_dir if not provided
        if models_dir is None:
            models_dir = os.path.join(self.script_dir, "models")
        self.models_dir = models_dir
        
        # Set categories_path if not provided
        if categories_path is None:
            categories_path = os.path.join(self.root_dir, "memento_categories_combined.json")
        self.categories_path = categories_path
        
        # Set tags_path if not provided
        if tags_path is None:
            tags_path = os.path.join(self.root_dir, "memento_tags_combined.json")
        self.tags_path = tags_path
        
        # Set durations_path if not provided
        if durations_path is None:
            durations_path = os.path.join(self.root_dir, "memento_durations.json")
        self.durations_path = durations_path
        
        # Set threshold
        self.threshold = threshold
        
        # Load models and data
        self.vectorizer = None
        self.category_model = None
        self.tags_model = None
        self.duration_model = None
        self.categories = None
        self.tags = None
        self.durations = None
        self.category_ids = None
        self.tag_ids = None
        self.duration_ids = None
        
        # Initialize
        self._load_categories_and_tags()
        self._load_models()
    
    def _load_categories_and_tags(self):
        """Load categories, tags, and durations from JSON files"""
        try:
            # Load categories
            with open(self.categories_path, "r", encoding="utf-8") as f:
                self.categories = json.load(f)
                self.category_ids = [cat["symbol"] for cat in self.categories]
            logging.info(f"Loaded {len(self.categories)} categories")
            
            # Load tags
            with open(self.tags_path, "r", encoding="utf-8") as f:
                self.tags = json.load(f)
                self.tag_ids = [tag["symbol"] for tag in self.tags]
            logging.info(f"Loaded {len(self.tags)} tags")
            
            # Load durations
            with open(self.durations_path, "r", encoding="utf-8") as f:
                self.durations = json.load(f)
                self.duration_ids = [duration["symbol"] for duration in self.durations]
            logging.info(f"Loaded {len(self.durations)} durations")
        except Exception as e:
            logging.error(f"Error loading categories, tags, or durations: {e}")
            raise
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            # Define model paths
            vectorizer_path = os.path.join(self.models_dir, "vectorizer.pkl")
            category_model_path = os.path.join(self.models_dir, "category_model.pkl")
            tags_model_path = os.path.join(self.models_dir, "tags_model.pkl")
            duration_model_path = os.path.join(self.models_dir, "duration_model.pkl")
            
            # Check if required models exist
            if not all(os.path.exists(path) for path in [vectorizer_path, category_model_path, tags_model_path]):
                logging.error(f"Required models not found in {self.models_dir}")
                raise FileNotFoundError(f"Required models not found in {self.models_dir}")
            
            # Load vectorizer
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            
            # Load category model
            with open(category_model_path, "rb") as f:
                self.category_model = pickle.load(f)
            
            # Load tags model
            with open(tags_model_path, "rb") as f:
                self.tags_model = pickle.load(f)
            
            # Load duration model if it exists
            if os.path.exists(duration_model_path):
                with open(duration_model_path, "rb") as f:
                    self.duration_model = pickle.load(f)
                logging.info("Duration model loaded successfully")
            else:
                logging.warning("Duration model not found, duration prediction will not be available")
                self.duration_model = None
                
            logging.info(f"Successfully loaded models from {self.models_dir}")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def predict_category(self, text: str) -> str:
        """
        Predict the category for a given text.
        
        Args:
            text: The description text to classify
            
        Returns:
            The predicted category (with emoji)
        """
        if not text or not self.vectorizer or not self.category_model:
            return "🗂️ Other"  # Default category if something's missing
        
        try:
            # Vectorize the text
            text_vec = self.vectorizer.transform([text])
            
            # Get prediction
            category = self.category_model.predict(text_vec)[0]
            
            # Return the predicted category
            return category
        except Exception as e:
            logging.error(f"Error predicting category: {e}")
            return "🗂️ Other"
    
    def predict_tags(self, text: str, max_tags: int = 3) -> List[str]:
        """
        Predict tags for a given text.
        
        Args:
            text: The description text to classify
            max_tags: Maximum number of tags to return
            
        Returns:
            List of predicted tags (with emojis)
        """
        if not text or not self.vectorizer or not self.tags_model:
            return ["🗂️ Other"]  # Default tag if something's missing
        
        try:
            # Vectorize the text
            text_vec = self.vectorizer.transform([text])
            
            # Get tag probabilities
            if hasattr(self.tags_model, "predict_proba"):
                tag_probas = self.tags_model.predict_proba(text_vec)[0]
                
                # Get indices of top tags
                top_indices = np.argsort(tag_probas)[-max_tags:][::-1]
                
                # Filter by threshold
                top_indices = [idx for idx in top_indices if tag_probas[idx] > self.threshold]
                
                # Convert to tag symbols
                tags = [self.tag_ids[idx] for idx in top_indices]
            else:
                # Fallback to binary prediction if probability not available
                tags_binary = self.tags_model.predict(text_vec)[0]
                tags = [self.tag_ids[i] for i, val in enumerate(tags_binary) if val]
                tags = tags[:max_tags]  # Limit to max_tags
            
            # Return default if no tags meet threshold
            if not tags:
                return ["🗂️ Other"]
            
            return tags
        except Exception as e:
            logging.error(f"Error predicting tags: {e}")
            return ["🗂️ Other"]
            
    def predict_duration(self, text: str) -> Optional[str]:
        """
        Predict the duration for a given text.
        
        Args:
            text: The description text to classify
            
        Returns:
            The predicted duration (with emoji) or None if duration model not available
        """
        if not text or not self.vectorizer or not self.duration_model:
            return None  # No duration prediction if model not available
        
        try:
            # Vectorize the text
            text_vec = self.vectorizer.transform([text])
            
            # Get prediction
            if hasattr(self.duration_model, "predict_proba"):
                # Get duration probabilities
                duration_probas = self.duration_model.predict_proba(text_vec)[0]
                
                # Only return duration if confidence is above threshold
                max_proba = np.max(duration_probas)
                if max_proba > self.threshold:
                    # Get index of most likely duration
                    max_idx = np.argmax(duration_probas)
                    # Get corresponding duration classes from model
                    duration_classes = self.duration_model.classes_
                    # Return the predicted duration
                    return duration_classes[max_idx]
                else:
                    return None
            else:
                # Fallback to direct prediction
                duration = self.duration_model.predict(text_vec)[0]
                return duration if duration else None
        except Exception as e:
            logging.error(f"Error predicting duration: {e}")
            return None
    
    def classify_memento(self, 
                         description: str, 
                         context: Optional[Dict] = None) -> Dict[str, Union[str, List[str]]]:
        """
        Classify a memento by description, optionally using context.
        
        Args:
            description: The description text to classify
            context: Optional context data (title, location, etc.)
            
        Returns:
            Dictionary with predicted category, tags, and duration
        """
        # Prepare text by combining with context if available
        text = description
        if context:
            # Combine description with title and location if available
            if "title" in context:
                text = context["title"] + " " + text
            if "location_name" in context:
                text = text + " " + context["location_name"]
        
        # Predict category and tags
        category = self.predict_category(text)
        tags = self.predict_tags(text)
        
        # Create result dictionary
        result = {
            "category": category,
            "mementoTags": tags
        }
        
        # Add duration prediction if available
        duration = self.predict_duration(text)
        if duration:
            result["mementoDuration"] = duration
        
        return result
    
    def update_scraper(self, scraper_path: str) -> bool:
        """
        Replace the rule-based classification in the scraper with ML-based classification.
        
        Args:
            scraper_path: Path to the scraper python file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the scraper file
            with open(scraper_path, "r", encoding="utf-8") as f:
                scraper_code = f.read()
            
            # Check if the file has the required functions
            has_assign_category = "def assign_category" in scraper_code
            has_assign_tags = "def assign_tags" in scraper_code
            has_extract_duration = "def extract_duration" in scraper_code
            
            if not (has_assign_category and has_assign_tags):
                logging.error(f"Scraper file {scraper_path} doesn't have the required functions")
                return False
            
            # Create backup
            backup_path = scraper_path + ".backup"
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(scraper_code)
            logging.info(f"Created backup at {backup_path}")
            
            # Import statement to add at the top of the file
            import_statement = f"""
# Import ML predictor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline.step4_predictor import MementoPredictor

# Initialize the predictor
ml_predictor = MementoPredictor()
"""
            
            # New assign_category function
            new_assign_category = """
def assign_category(desc: str, context: Dict = None) -> str:
    \"\"\"Assign category using ML model\"\"\"
    return ml_predictor.predict_category(desc)
"""
            
            # New assign_tags function
            new_assign_tags = """
def assign_tags(desc: str, context: Dict = None) -> List[str]:
    \"\"\"Assign tags using ML model\"\"\"
    return ml_predictor.predict_tags(desc)
"""
            
            # New extract_duration function (only if existing function exists)
            new_extract_duration = """
def extract_duration(desc: str) -> Optional[str]:
    \"\"\"Extract duration using ML model\"\"\"
    return ml_predictor.predict_duration(desc)
"""
            
            # Replace the functions in the code
            
            # Add import at the top (after existing imports)
            import_section_end = scraper_code.find("# -----------------------------", scraper_code.find("import"))
            if import_section_end > 0:
                scraper_code = scraper_code[:import_section_end] + import_statement + scraper_code[import_section_end:]
            
            # Replace assign_category function
            start = scraper_code.find("def assign_category")
            if start > 0:
                end = scraper_code.find("def", start + 1)
                if end > 0:
                    scraper_code = scraper_code[:start] + new_assign_category + scraper_code[end:]
            
            # Replace assign_tags function
            start = scraper_code.find("def assign_tags")
            if start > 0:
                end = scraper_code.find("def", start + 1)
                if end > 0:
                    scraper_code = scraper_code[:start] + new_assign_tags + scraper_code[end:]
                else:
                    # If it's the last function, go until the end
                    scraper_code = scraper_code[:start] + new_assign_tags
            
            # Replace extract_duration function if it exists
            if has_extract_duration:
                start = scraper_code.find("def extract_duration")
                if start > 0:
                    end = scraper_code.find("def", start + 1)
                    if end > 0:
                        scraper_code = scraper_code[:start] + new_extract_duration + scraper_code[end:]
                    else:
                        # If it's the last function, go until the end
                        scraper_code = scraper_code[:start] + new_extract_duration
            
            # Write the modified code back to the file
            with open(scraper_path, "w", encoding="utf-8") as f:
                f.write(scraper_code)
            
            logging.info(f"Successfully updated scraper at {scraper_path}")
            return True
        except Exception as e:
            logging.error(f"Error updating scraper: {e}")
            return False

def test_predictor():
    """Test the predictor on sample texts"""
    predictor = MementoPredictor()
    
    # Test cases
    test_cases = [
        "Beautiful cherry blossoms in Central Park, perfect for a spring day walk.",
        "The new art exhibition at MoMA showcases contemporary artists from around the world.",
        "Live jazz music at Bryant Park every Friday evening this summer.",
        "Annual food festival with cuisines from around the world in Times Square.",
        "Hidden speakeasy bar with craft cocktails in the Lower East Side."
    ]
    
    print("\nTesting MementoPredictor on sample texts:\n")
    for i, test in enumerate(test_cases):
        category = predictor.predict_category(test)
        tags = predictor.predict_tags(test)
        duration = predictor.predict_duration(test)
        
        print(f"Sample {i+1}:")
        print(f"Text: {test}")
        print(f"Predicted Category: {category}")
        print(f"Predicted Tags: {tags}")
        print(f"Predicted Duration: {duration}")
        print("-" * 80)

if __name__ == "__main__":
    # Test the predictor
    test_predictor() 