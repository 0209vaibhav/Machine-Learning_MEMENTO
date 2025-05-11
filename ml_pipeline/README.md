# ML Pipeline

This pipeline processes user mementos and scraped data through ML models to classify them into categories, tags, and durations.

## Setup

### Firebase Credentials Setup

1. Go to the [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Go to Project Settings > Service Accounts
4. Click "Generate New Private Key"
5. Save the downloaded JSON file as `firebase-credentials.json` in the `ml_pipeline` directory
6. Make sure the file is not tracked by git (it's already in .gitignore)

Note: Never commit your `firebase-credentials.json` file to version control. Use the provided `firebase-credentials.template.json` as a reference for the required format.

## Pipeline Steps

1. **Data Loading** (`step1_data_loader.py`)
   - Loads user mementos from Firebase
   - Preprocesses text data
   - Outputs: `output/step1_data_processing/user_mementos_processed.csv`

2. **Model Architecture** (`step2_model.py`)
   - Defines ML model architectures for:
     - Category classification
     - Tag prediction
     - Duration estimation
   - Outputs: Model class definitions

3. **Model Training** (`step3_train_models.py`)
   - Trains ML models using user mementos
   - Uses processed data from step1
   - Outputs trained models to `output/step3_model_training/`

4. **Data Scraping** (`step4_scrape_data.py`)
   - Scrapes data from Secret NYC
   - Collects mementos to be classified
   - Outputs: `output/step4_scraped_data/scraped_mementos.json`

5. **Data Processing** (`step5_process_scraped.py`)
   - Processes scraped data through trained models
   - Uses models from step3
   - Outputs: `output/step5_processed_data/classified_mementos.csv`

6. **Prediction Service** (`step6_predictor.py`)
   - Makes predictions available through predictor
   - Provides API for memento classification
   - Outputs: Prediction service endpoints

7. **System Integration** (`step7_integration.py`)
   - Integrates results back into system
   - Updates Firebase with classified mementos
   - Handles UI integration

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data:
```bash
python download_nltk_data.py
```

3. Run the pipeline:
```bash
python run_pipeline.py [options]
```

Available options:
- `--skip-scraping`: Skip step 4 (scraping data)
- `--skip-training`: Skip steps 2-3 (model training)
- `--skip-processing`: Skip step 5 (processing scraped data)
- `--force-retrain`: Force retraining of models
- `--debug`: Enable debug logging

## Directory Structure

```
ml_pipeline/
├── step1_data_loader.py
├── step2_model.py
├── step3_train_models.py
├── step4_scrape_data.py
├── step5_process_scraped.py
├── step6_predictor.py
├── step7_integration.py
├── run_pipeline.py
├── download_nltk_data.py
├── requirements.txt
├── README.md
└── output/
    ├── step1_data_processing/
    ├── step3_model_training/
    ├── step4_scraped_data/
    ├── step5_processed_data/
    ├── step6_predictions/
    └── step7_integration/
```

## Requirements

- Python 3.8+
- Firebase credentials (`firebase-credentials.json`)
- Internet connection for scraping
- NLTK data for text processing

## Notes

- User mementos from Firebase are used as training data
- Scraped data from Secret NYC is used for testing/classification
- Models are trained on real user data to ensure quality predictions
- Integration with the main system happens in step 7 