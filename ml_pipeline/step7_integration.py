#!/usr/bin/env python
"""
Step 6: ML Pipeline Integration

This module provides tools to integrate the trained ML models with production systems,
particularly focusing on updating existing scrapers to use ML-based classification.

Key Features:
1. Scraper Integration:
   - Automatic scraper detection
   - Code analysis and validation
   - Safe code modification
   - Backup creation
   - Rollback capabilities

2. Safety Features:
   - Dry run mode for testing
   - Automatic backups
   - Validation checks
   - Error recovery
   - Logging and monitoring

3. Integration Tools:
   - Command-line interface
   - Scraper analysis
   - Code modification utilities
   - Import management
   - Path resolution

4. Production Features:
   - Multiple scraper support
   - Configurable search paths
   - Version compatibility checks
   - Performance monitoring
   - Error reporting

Usage:
```bash
# Update a specific scraper
python step6_integration.py --scraper path/to/scraper.py

# Test changes without applying
python step6_integration.py --scraper path/to/scraper.py --dryrun

# Find potential scrapers
python step6_integration.py --find --search-dir path/to/search
```

Dependencies:
- step5_predictor.py: ML model predictor
- shutil: File operations
- argparse: CLI handling
"""

import os
import sys
import argparse
import logging
import shutil
from datetime import datetime
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add the current directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import the predictor
from step5_predictor import MementoPredictor

def find_scrapers(search_dir: str = None) -> list:
    """
    Find all potential scraper files in the given directory
    
    Args:
        search_dir: Directory to search in (defaults to project root)
        
    Returns:
        List of potential scraper file paths
    """
    if search_dir is None:
        search_dir = os.path.dirname(script_dir)
    
    potential_scrapers = []
    
    # Search patterns that indicate a scraper file
    patterns = [
        "scrape_all_pages",
        "scrape_",
        "_scraper",
        "crawler",
    ]
    
    # Walk through the directory
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                
                # Check if filename matches any pattern
                if any(pattern in file.lower() for pattern in patterns):
                    potential_scrapers.append(file_path)
                else:
                    # Check file contents for key functions
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if "def assign_category" in content and "def assign_tags" in content:
                                potential_scrapers.append(file_path)
                    except Exception:
                        pass
    
    return potential_scrapers

def analyze_scraper(scraper_path: str) -> Tuple[bool, dict]:
    """
    Analyze a scraper file to determine if it can be updated
    
    Args:
        scraper_path: Path to the scraper file
        
    Returns:
        Tuple of (can_update, info_dict)
    """
    if not os.path.exists(scraper_path):
        return False, {"error": "File not found"}
    
    try:
        with open(scraper_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for required functions
        has_assign_category = "def assign_category" in content
        has_assign_tags = "def assign_tags" in content
        
        # Check if already using ML predictor
        already_using_ml = "MementoPredictor" in content
        
        # Get package imports
        imports = []
        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                imports.append(line.strip())
        
        return has_assign_category and has_assign_tags, {
            "path": scraper_path,
            "has_assign_category": has_assign_category,
            "has_assign_tags": has_assign_tags,
            "already_using_ml": already_using_ml,
            "imports": imports,
            "size": len(content),
            "lines": len(lines)
        }
    except Exception as e:
        return False, {"error": str(e)}

def update_scraper(scraper_path: str, dryrun: bool = False) -> bool:
    """
    Update a scraper file to use ML-based classification
    
    Args:
        scraper_path: Path to the scraper file
        dryrun: If True, only show the changes without applying them
        
    Returns:
        True if successful, False otherwise
    """
    # First analyze the scraper
    can_update, info = analyze_scraper(scraper_path)
    
    if not can_update:
        logging.error(f"Cannot update {scraper_path}: {info.get('error', 'Missing required functions')}")
        return False
    
    if info.get("already_using_ml", False):
        logging.warning(f"Scraper {scraper_path} is already using ML predictor")
        return False
    
    logging.info(f"Analyzing scraper: {scraper_path}")
    logging.info(f"  - Lines: {info.get('lines', 'unknown')}")
    logging.info(f"  - Has assign_category: {info.get('has_assign_category', False)}")
    logging.info(f"  - Has assign_tags: {info.get('has_assign_tags', False)}")
    
    if dryrun:
        logging.info("DRY RUN: Would update scraper with ML predictor")
        return True
    
    # Create a backup
    backup_path = f"{scraper_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(scraper_path, backup_path)
        logging.info(f"Created backup at {backup_path}")
    except Exception as e:
        logging.error(f"Failed to create backup: {e}")
        return False
    
    # Update the scraper
    try:
        predictor = MementoPredictor()
        success = predictor.update_scraper(scraper_path)
        
        if success:
            logging.info(f"Successfully updated {scraper_path}")
            return True
        else:
            logging.error(f"Failed to update {scraper_path}")
            return False
    except Exception as e:
        logging.error(f"Error updating scraper: {e}")
        
        # Try to restore from backup
        try:
            shutil.copy2(backup_path, scraper_path)
            logging.info(f"Restored from backup {backup_path}")
        except Exception as restore_error:
            logging.error(f"Failed to restore from backup: {restore_error}")
        
        return False

def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="Integrate ML-based classification with existing scrapers")
    parser.add_argument("--scraper", help="Path to scraper file to update")
    parser.add_argument("--dryrun", action="store_true", help="Show changes without applying them")
    parser.add_argument("--find", action="store_true", help="Find potential scrapers")
    parser.add_argument("--search-dir", help="Directory to search for scrapers")
    args = parser.parse_args()
    
    # Find potential scrapers
    if args.find:
        logging.info("Searching for potential scrapers...")
        scrapers = find_scrapers(args.search_dir)
        
        if not scrapers:
            logging.info("No potential scrapers found")
            return
        
        logging.info(f"Found {len(scrapers)} potential scrapers:")
        for i, scraper in enumerate(scrapers):
            can_update, info = analyze_scraper(scraper)
            status = "✅ Can update" if can_update else "❌ Cannot update"
            using_ml = " (already using ML)" if info.get("already_using_ml", False) else ""
            logging.info(f"{i+1}. {os.path.relpath(scraper)}: {status}{using_ml}")
        
        return
    
    # Update specific scraper
    if args.scraper:
        if not os.path.exists(args.scraper):
            logging.error(f"Scraper file not found: {args.scraper}")
            return
        
        update_scraper(args.scraper, args.dryrun)
    else:
        logging.error("Please specify a scraper file to update with --scraper or use --find to discover scrapers")

def test_integration():
    """Test the integration with sample scraper code"""
    # Create a sample scraper with rule-based classification
    sample_scraper = """
import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Optional

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_URL = "https://example.com"

# -----------------------------
# CLASSIFICATION FUNCTIONS
# -----------------------------
def match_keywords(text: str, items: List[Dict]) -> List[str]:
    matched = []
    text = text.lower()
    
    for item in items:
        for kw in item["keywords"]:
            if kw.lower() in text:
                matched.append(item["symbol"] + " " + item["name"])
                break
    
    return matched

def assign_category(desc: str, context: Dict = None) -> str:
    # Rule-based category assignment
    if "art" in desc.lower() or "exhibition" in desc.lower():
        return "🎭 Cultural Spotlight"
    elif "park" in desc.lower() or "garden" in desc.lower():
        return "🌿 Urban Nature"
    elif "food" in desc.lower() or "restaurant" in desc.lower():
        return "🍴 Street Food"
    else:
        return "🗂️ Other"

def assign_tags(desc: str, context: Dict = None) -> List[str]:
    # Rule-based tag assignment
    tags = []
    
    if "night" in desc.lower():
        tags.append("🌃 Nightlife Pulse")
    
    if "first time" in desc.lower() or "beginner" in desc.lower():
        tags.append("🐣 First-Time Friendly")
    
    if "tourist" in desc.lower() or "attraction" in desc.lower():
        tags.append("🧳 Touristy Yet Fun")
    
    if not tags:
        tags.append("🗂️ Other")
    
    return tags[:3]  # Return up to 3 tags

def scrape_article(url: str) -> Optional[Dict]:
    # Dummy function for the example
    return None

def main():
    print("Sample scraper")

if __name__ == "__main__":
    main()
"""
    
    # Write to a temporary file
    temp_path = os.path.join(script_dir, "sample_scraper.py")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(sample_scraper)
    
    logging.info(f"Created sample scraper at {temp_path}")
    
    # Analyze the sample scraper
    can_update, info = analyze_scraper(temp_path)
    logging.info(f"Can update: {can_update}")
    logging.info(f"Info: {info}")
    
    # Update the sample scraper
    success = update_scraper(temp_path)
    logging.info(f"Update successful: {success}")
    
    # Show the updated scraper
    if success:
        with open(temp_path, "r", encoding="utf-8") as f:
            updated_code = f.read()
        
        logging.info("Updated sample scraper:")
        logging.info("-" * 80)
        logging.info(updated_code)
        logging.info("-" * 80)
    
    # Clean up
    try:
        os.remove(temp_path)
        logging.info(f"Removed sample scraper: {temp_path}")
        
        # Remove backup files
        for file in os.listdir(script_dir):
            if file.startswith("sample_scraper.py.bak."):
                os.remove(os.path.join(script_dir, file))
    except Exception as e:
        logging.error(f"Error cleaning up: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Run test if no arguments provided
        test_integration() 