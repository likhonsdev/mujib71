import requests
from bs4 import BeautifulSoup
import time
import random
import json
from pathlib import Path
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BengaliDataCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.output_dir = Path('data/raw')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def make_request(self, url, retries=3, delay=1):
        """Make HTTP request with retry logic and rate limiting"""
        for attempt in range(retries):
            try:
                time.sleep(delay + random.random())  # Rate limiting with jitter
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == retries - 1:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    raise
                time.sleep(delay * (attempt + 1))  # Exponential backoff
                
    def scrape_wikipedia(self):
        """Scrape Bengali text from Wikipedia"""
        url = "https://bn.wikipedia.org/wiki/প্রধান_পাতা"
        logger.info(f"Scraping Wikipedia: {url}")
        
        try:
            response = self.make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get main content and featured articles
            content_div = soup.find('div', {'id': 'mw-content-text'})
            articles = []
            
            if content_div:
                # Extract article links
                article_links = content_div.find_all('a', href=True)
                for link in article_links[:50]:  # Limit to first 50 articles
                    if link['href'].startswith('/wiki/') and ':' not in link['href']:
                        article_url = urljoin('https://bn.wikipedia.org', link['href'])
                        try:
                            article_response = self.make_request(article_url)
                            article_soup = BeautifulSoup(article_response.content, 'html.parser')
                            
                            # Extract article content
                            article_content = article_soup.find('div', {'id': 'mw-content-text'})
                            if article_content:
                                text = article_content.get_text(separator='\n', strip=True)
                                articles.append({
                                    'url': article_url,
                                    'content': text
                                })
                                logger.info(f"Successfully scraped article: {article_url}")
                        except Exception as e:
                            logger.error(f"Failed to scrape article {article_url}: {str(e)}")
                            
            # Save Wikipedia data
            with open(self.output_dir / 'wikipedia_data.json', 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
                
            return len(articles)
        except Exception as e:
            logger.error(f"Failed to scrape Wikipedia: {str(e)}")
            return 0

    def scrape_prothom_alo(self):
        """Scrape Bengali text from Prothom Alo"""
        base_url = "https://www.prothomalo.com"
        categories = ['bangladesh', 'international', 'opinion', 'science-technology']
        articles = []
        
        for category in categories:
            url = f"{base_url}/{category}"
            logger.info(f"Scraping Prothom Alo category: {category}")
            
            try:
                response = self.make_request(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                article_links = soup.find_all('a', href=True)
                for link in article_links[:10]:  # Limit to 10 articles per category
                    article_url = urljoin(base_url, link['href'])
                    if category in article_url:
                        try:
                            article_response = self.make_request(article_url)
                            article_soup = BeautifulSoup(article_response.content, 'html.parser')
                            
                            # Extract article content
                            article_content = article_soup.find('div', {'class': 'story-content'})
                            if article_content:
                                text = article_content.get_text(separator='\n', strip=True)
                                articles.append({
                                    'url': article_url,
                                    'category': category,
                                    'content': text
                                })
                                logger.info(f"Successfully scraped article: {article_url}")
                        except Exception as e:
                            logger.error(f"Failed to scrape article {article_url}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Failed to scrape category {category}: {str(e)}")
                
        # Save Prothom Alo data
        with open(self.output_dir / 'prothomalo_data.json', 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
            
        return len(articles)

    def collect(self):
        """Main method to collect data from all sources"""
        logger.info("Starting data collection")
        
        wiki_count = self.scrape_wikipedia()
        logger.info(f"Collected {wiki_count} articles from Wikipedia")
        
        prothomalo_count = self.scrape_prothom_alo()
        logger.info(f"Collected {prothomalo_count} articles from Prothom Alo")
        
        # Combine and process the collected data
        self.process_collected_data()
        
        logger.info("Data collection completed")
        
    def process_collected_data(self):
        """Process and combine collected data"""
        try:
            # Read collected data
            with open(self.output_dir / 'wikipedia_data.json', 'r', encoding='utf-8') as f:
                wiki_data = json.load(f)
            
            with open(self.output_dir / 'prothomalo_data.json', 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            
            # Combine and format data
            processed_data = []
            
            # Process Wikipedia articles
            for article in wiki_data:
                processed_data.append({
                    'text': article['content'],
                    'source': 'wikipedia',
                    'url': article['url']
                })
            
            # Process news articles
            for article in news_data:
                processed_data.append({
                    'text': article['content'],
                    'source': 'prothomalo',
                    'category': article.get('category', ''),
                    'url': article['url']
                })
            
            # Save processed data
            with open(self.output_dir / 'processed_data.json', 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Successfully processed {len(processed_data)} articles")
            
        except Exception as e:
            logger.error(f"Failed to process collected data: {str(e)}")
            raise

if __name__ == "__main__":
    collector = BengaliDataCollector()
    collector.collect()
