#!/usr/bin/env python
"""
ML Pipeline Runner

This script orchestrates the execution of the ML pipeline in the following sequence:
1. Load user mementos from Firebase (training data)
2. Define ML model architectures
3. Train models using user mementos
4. Scrape data from Secret NYC
5. Process scraped data through trained models
6. Make predictions available through predictor
7. Integrate results back into system

Usage:
    python run_pipeline.py [options]

Options:
    --skip-scraping: Skip step 4 (scraping data)
    --skip-training: Skip steps 2-3 (model training)
    --skip-processing: Skip step 5 (processing scraped data)
    --force-retrain: Force retraining of models even if they exist
    --debug: Enable debug logging

Example:
    python run_pipeline.py --skip-scraping --force-retrain
"""

import os
import sys
import logging
import argparse
from datetime import datetime

def setup_logging(debug=False):
    """Set up logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"pipeline_run_{timestamp}.log"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def setup_directories():
    """Create necessary output directories"""
    dirs = [
        "output/step1_data_processing",
        "output/step3_model_training",
        "output/step4_scraped_data",
        "output/step5_processed_data",
        "output/step6_predictions",
        "output/step7_integration"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the ML pipeline")
    parser.add_argument("--skip-scraping", action="store_true", help="Skip scraping data")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-processing", action="store_true", help="Skip processing scraped data")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def run_step1_data_loading():
    """Step 1: Load user mementos from Firebase"""
    logging.info("Step 1: Loading user mementos from Firebase")
    from step1_data_loader import main as data_loader_main
    data_loader_main()

def run_step2_model_definition():
    """Step 2: Define ML model architectures"""
    logging.info("Step 2: Defining ML model architectures")
    from step2_model import main as model_main
    model_main()

def run_step3_model_training():
    """Step 3: Train models using user mementos"""
    logging.info("Step 3: Training ML models")
    from step3_train_models import main as train_main
    train_main()

def run_step4_scraping():
    """Step 4: Scrape data from Secret NYC"""
    logging.info("Step 4: Scraping data from Secret NYC")
    from step4_scrape_data import main as scrape_main
    scrape_main()

def run_step5_processing():
    """Step 5: Process scraped data through trained models"""
    logging.info("Step 5: Processing scraped data")
    from step5_process_scraped import main as process_main
    process_main()

def run_step6_prediction():
    """Step 6: Make predictions available"""
    logging.info("Step 6: Setting up prediction service")
    from step6_predictor import main as predictor_main
    predictor_main()

def run_step7_integration():
    """Step 7: Integrate results back into system"""
    logging.info("Step 7: Integrating results")
    from step7_integration import main as integration_main
    integration_main()

def main():
    """Main function to run the pipeline"""
    args = parse_args()
    log_file = setup_logging(args.debug)
    logging.info(f"Starting ML pipeline. Logs will be saved to: {log_file}")
    
    try:
        # Create output directories
        setup_directories()
        
        # Step 1: Always load data
        run_step1_data_loading()
        
        # Steps 2-3: Model definition and training
        if not args.skip_training:
            run_step2_model_definition()
            run_step3_model_training()
        else:
            logging.info("Skipping model training (steps 2-3)")
        
        # Step 4: Scraping
        if not args.skip_scraping:
            run_step4_scraping()
        else:
            logging.info("Skipping data scraping (step 4)")
        
        # Step 5: Processing
        if not args.skip_processing:
            run_step5_processing()
        else:
            logging.info("Skipping data processing (step 5)")
        
        # Step 6: Always set up prediction service
        run_step6_prediction()
        
        # Step 7: Always run integration
        run_step7_integration()
        
        logging.info("ML pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 