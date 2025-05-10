"""
Step 4: Scrape Data

This module scrapes data from Secret NYC to create a testing dataset.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
from datetime import datetime
from langdetect import detect
from typing import Dict, List, Optional
import logging
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class SecretNYCScraper:
    """Scrapes data from Secret NYC website"""
    
    def __init__(self, 
                 base_url: str = "https://secretnyc.co/things-to-do/",
                 output_dir: str = "ml_pipeline/output/step4_scraped_data"):
        """Initialize scraper"""
        self.base_url = base_url
        self.output_dir = output_dir
        self.raw_data_dir = os.path.join(output_dir, "raw_data")
        
        # Create output directories
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # Default user
        self.default_user = "Secret NYC"
    
    def create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        return session
        
    def get_article_links(self) -> List[str]:
        """Get article links from main page"""
        session = self.create_session()
        try:
            response = session.get(self.base_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if (
                    href.startswith("https://secretnyc.co/")
                    and "/page/" not in href
                    and "/author/" not in href
                    and "/category/" not in href
                    and "#" not in href
                    and href.count("/") > 3
                ):
                    links.append(href)
            return list(set(links))
        except Exception as e:
            logging.error(f"Error fetching article links: {e}")
            return []
    
    def geocode_location(self, query: str) -> Optional[Dict[str, float]]:
        """Get coordinates for location using OpenStreetMap"""
        if not query:
            return None
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": query + ", New York City", "format": "json", "limit": 1}
            headers = {"User-Agent": "MEMENTO-map-collector/1.0"}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data:
                return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
        except Exception as e:
            logging.error(f"Error geocoding location {query}: {e}")
            return None
            
    def extract_fallback_location(self, description: str, title: str) -> Optional[str]:
        """Extract location from text using regex patterns"""
        combined_text = title + " " + description
        patterns = [
            r"(?:at|in|near|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(Avenue|Street|Boulevard|Park|Square|Bridge)"
        ]
        for pattern in patterns:
            match = re.search(pattern, combined_text)
            if match:
                return match.group(1).strip()
        return None
            
    def clean_description(self, text: str) -> str:
        """Clean and format description text"""
        if not text:
            return ""
        text = re.sub(r"A post shared by.*?(\n|$)", "", text)
        text = re.sub(r"This year.*?website reads:", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([.!?])\s*", r"\1 ", text).strip()
        return " ".join(re.split(r'(?<=[.!?])\s+', text)[:5])

    def parse_date(self, date_str: str) -> str:
        """Parse date string to consistent format"""
        try:
            formats = [
                "%Y-%m-%dT%H:%M:%S%z",
                "%B %d, %Y",
                "%A, %B %d, %Y",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip(), fmt)
                    return dt.strftime("%B %d, %Y at %I:%M %p")
                except ValueError:
                    continue
            match = re.search(r'(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)?,?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}', date_str)
            if match:
                try:
                    dt = datetime.strptime(match.group(0), "%A, %B %d, %Y")
                    return dt.strftime("%B %d, %Y at %I:%M %p")
                except ValueError:
                    pass
            return datetime.now().strftime("%B %d, %Y at %I:%M %p")
        except:
            return datetime.now().strftime("%B %d, %Y at %I:%M %p")

    def extract_duration(self, text: str) -> str:
        """Extract duration from text using regex patterns"""
        patterns = [
            r"(?:takes|lasts|duration|time|spend|spent)\s+(?:about|around|approximately|roughly)?\s*(\d+)\s*(?:hour|hr|minute|min|day|week|month)s?",
            r"(\d+)\s*(?:hour|hr|minute|min|day|week|month)s?\s+(?:long|duration|time)",
            r"(\d+)\s*(?:hour|hr|minute|min|day|week|month)s?\s+(?:to|for|in)\s+(?:complete|finish|visit|explore|experience)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                duration = match.group(1)
                # Convert to standard format
                if "hour" in text.lower() or "hr" in text.lower():
                    return f"{duration}h"
                elif "minute" in text.lower() or "min" in text.lower():
                    return f"{duration}m"
                elif "day" in text.lower():
                    return f"{duration}d"
                elif "week" in text.lower():
                    return f"{duration}w"
                elif "month" in text.lower():
                    return f"{duration}mo"
        return "1h"  # Default duration if not found

    def scrape_article(self, article_url: str) -> Optional[Dict]:
        """Scrape a single article"""
        session = self.create_session()
        try:
            response = session.get(article_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "Untitled"
            
            # Extract description
            paragraphs = soup.select("section.article__body p")
            desc = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            
            # Language check
            try:
                if detect(desc) != "en":
                    return None
            except:
                return None
            
            cleaned = self.clean_description(desc)
            
            # Extract media
            img_tag = soup.select_one("section.article__body figure img")
            media_url = img_tag["src"] if img_tag and img_tag.has_attr("src") else None
            
            # Extract author
            author = soup.select_one("section.article__body figure figcaption")
            author = author.get_text(strip=True) if author else self.default_user
            if author != self.default_user and not author.startswith("Source /"):
                author = f"Source / {author}"
            
            # Extract date
            date_line = next((p.get_text(strip=True) for p in paragraphs if "🗓️" in p.get_text()), None)
            time_tag = soup.find("time")
            timestamp = (
                self.parse_date(date_line.replace("🗓️", "").strip()) if date_line else
                self.parse_date(time_tag["datetime"]) if time_tag and time_tag.has_attr("datetime") else
                datetime.now().strftime("%B %d, %Y at %I:%M %p")
            )
            
            # Extract location
            loc_line = next((p.get_text(strip=True) for p in paragraphs if "📍" in p.get_text()), None)
            loc_name = loc_line.replace("📍", "").strip() if loc_line else self.extract_fallback_location(desc, title)
            coords = self.geocode_location(loc_name) if loc_name else None
            
            # Extract duration
            duration = self.extract_duration(desc)
            
            # Create memento
            memento = {
                "userId": author,
                "location": coords if coords else {},
                "media": [media_url] if media_url and not media_url.startswith("data:image/svg") else [],
                "name": title,
                "description": cleaned,
                "category": "Other",  # Will be classified by ML
                "timestamp": timestamp,
                "tags": ["Other"],  # Will be classified by ML
                "link": article_url,
                "mementoType": "public",
                "duration": duration  # Add duration field
            }
            
            return memento
            
        except Exception as e:
            logging.error(f"Error scraping {article_url}: {e}")
            return None
    
    def scrape_data(self) -> List[Dict]:
        """Scrape data from Secret NYC"""
        logging.info("Starting to scrape Secret NYC...")
        
        # Get article links
        article_links = self.get_article_links()
        logging.info(f"Found {len(article_links)} articles")
        
        # Scrape articles
        mementos = []
        for i, url in enumerate(article_links, 1):
            try:
                memento = self.scrape_article(url)
                if memento:
                    mementos.append(memento)
                logging.info(f"[{i}/{len(article_links)}] Processed {url}")
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
            
            # Add delay between requests
            time.sleep(2)
        
        return mementos

    def save_data(self, mementos: List[Dict]):
        """Save scraped data"""
        output_path = os.path.join(self.raw_data_dir, "scraped_mementos.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(mementos, f, indent=4, ensure_ascii=False)
            logging.info(f"Saved {len(mementos)} mementos to {output_path}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")

def main():
    """Main function"""
    # Initialize scraper
    scraper = SecretNYCScraper()
    
    # Scrape data
    mementos = scraper.scrape_data()
    
    # Save data
    scraper.save_data(mementos)

if __name__ == "__main__":
    main() 