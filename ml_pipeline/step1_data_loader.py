"""
Step 1: Data Loading and Preparation

This module is responsible for loading user mementos from Firebase for ML training.
These user mementos will be used as training data for the ML models.

Key Features:
1. Data Loading:
   - Loads user mementos from Firebase
   - Loads categories, tags, and duration metadata
   - No filtering by mementoType (all mementos are used)

2. Data Processing:
   - Cleans and normalizes text data
   - Extracts features for ML training
   - Handles missing values and data validation

3. Data Preparation:
   - Converts text to ML-ready format
   - Prepares data for model training
   - Saves processed data for next steps

Output:
- Processed DataFrame saved to: output/step1_data_processing/user_mementos_processed.csv
  Contains:
  - Text features for ML
  - Categories
  - Tags
  - Durations
  - Additional metadata (location, timestamps, etc.)

Dependencies:
- pandas: Data manipulation
- firebase_admin: Firebase integration
- nltk: Text processing
"""

import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.preprocessing import MultiLabelBinarizer
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class MementoDataLoader:
    """
    Step 1: Data Preparation
    
    This class handles loading user memento data from Firebase,
    preparing it for training machine learning models for categorization,
    tagging, and duration prediction.
    """
    
    def __init__(self, 
                 categories_path: str = None,
                 tags_path: str = None,
                 durations_path: str = None,
                 firebase_credentials_path: str = None):
        """Initialize the data loader"""
        self.categories = {}
        self.tags = {}
        self.durations = {}
        self.db = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load metadata if paths provided
        if categories_path:
            self.load_categories(categories_path)
        if tags_path:
            self.load_tags(tags_path)
        if durations_path:
            self.load_durations(durations_path)
            
        # Initialize Firebase if credentials provided
        if firebase_credentials_path:
            self.firebase_credentials_path = firebase_credentials_path
            self._initialize_firebase()
        
    def load_categories(self, categories_path: str):
        """Load categories from JSON file"""
        logging.info(f"Loading categories from {categories_path}")
        with open(categories_path, "r", encoding="utf-8") as f:
            self.categories = json.load(f)
        logging.info(f"Loaded {len(self.categories)} categories")
        
    def load_tags(self, tags_path: str) -> List[str]:
        """Load tags from JSON file"""
        logging.info(f"Loading tags from {tags_path}")
        with open(tags_path, "r", encoding="utf-8") as f:
            tags_data = json.load(f)
            tags = [tag["name"] for tag in tags_data]
        logging.info(f"Loaded {len(tags)} tags")
        return tags
        
    def load_durations(self, durations_path: str):
        """Load durations from JSON file"""
        logging.info(f"Loading durations from {durations_path}")
        with open(durations_path, "r", encoding="utf-8") as f:
            self.durations = json.load(f)
        logging.info(f"Loaded {len(self.durations)} durations")
        
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            cred = credentials.Certificate(self.firebase_credentials_path)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logging.info("Firebase initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Firebase: {e}")
    
    def load_from_firebase(self, limit=1000):
        """Load user mementos from Firebase"""
        try:
            logging.info("Loading mementos from Firebase...")
            logging.info(f"Loading up to {limit} mementos from Firebase")
            
            # Query all mementos from Firestore
            mementos_ref = self.db.collection("mementos").limit(limit)
            logging.info("Executing Firebase query: all mementos")
            
            # Get the documents
            mementos = mementos_ref.get()
            logging.info(f"Found {len(mementos)} mementos in Firebase")
            
            # Convert to list of dictionaries
            memento_list = []
            for doc in mementos:
                memento_data = doc.to_dict()
                memento_data['id'] = doc.id
                memento_list.append(memento_data)
            
            logging.info(f"Processed {len(memento_list)} mementos from Firebase")
            
            if not memento_list:
                raise ValueError("No mementos found in Firebase")
            
            return pd.DataFrame(memento_list)
            
        except Exception as e:
            logging.error(f"Error loading mementos from Firebase: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for ML by:
        1. Converting to lowercase
        2. Removing special characters
        3. Removing extra whitespace
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def _extract_text_for_ml(self, memento: Dict) -> str:
        """
        Extract and preprocess text for ML from memento.
        
        Args:
            memento: Dictionary containing memento data
            
        Returns:
            Preprocessed text for ML
        """
        text_parts = []
        
        # Add name with higher weight (repeat 3 times)
        name = memento.get('name', '')
        text_parts.extend([name] * 3)
        
        # Add description
        description = memento.get('description', '')
        text_parts.append(description)
        
        # Add location name if available
        location = memento.get('location', {})
        if isinstance(location, dict):
            location_str = str(location)
            text_parts.append(location_str)
        
        # Join all parts and preprocess
        combined_text = ' '.join(filter(None, text_parts))
        return self._preprocess_text(combined_text)

    def prepare_training_data(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """
        Prepare data for training by processing user mementos.
        
        Args:
            df: DataFrame with raw memento data
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary with paths to saved files
        """
        logging.info("Preparing data for training...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the data
        processed_data = []
        for _, memento in df.iterrows():
            memento_dict = memento.to_dict()
            
            # Extract and preprocess text
            text_for_ml = self._extract_text_for_ml(memento_dict)
            
            # Add processed text to memento
            memento_dict['text_for_ml'] = text_for_ml
            
            processed_data.append(memento_dict)
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        # Save processed data
        output_path = os.path.join(output_dir, "user_mementos_processed.csv")
        processed_df.to_csv(output_path, index=False)
        
        logging.info(f"Saved processed data to {output_path}")
        
        return {
            "processed_data": output_path
        }

def main():
    """Main function"""
    # Initialize data loader with paths
    loader = MementoDataLoader(
        categories_path="memento_categories.json",
        tags_path="memento_tags.json",
        durations_path="memento_durations.json",
        firebase_credentials_path="config/serviceAccountKey.json"
    )
    
    # Create output directory if it doesn't exist
    output_dir = "ml_pipeline/output/step1_data_processing/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load user mementos from Firebase
    logging.info("Loading user mementos from Firebase...")
    user_mementos = loader.load_from_firebase(limit=1000)
    
    if user_mementos.empty:
        logging.error("No user mementos found in Firebase")
        return
    
    logging.info(f"Loaded {len(user_mementos)} user mementos from Firebase")
    
    # Process the data
    processed_data = loader.prepare_training_data(user_mementos, output_dir)
    
    # Save processed data
    output_file = os.path.join(output_dir, "user_mementos_processed.csv")
    user_mementos.to_csv(output_file, index=False)
    logging.info(f"Saved processed user mementos to {output_file}")

if __name__ == "__main__":
    main() 