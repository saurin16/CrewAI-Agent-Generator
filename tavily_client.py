from dotenv import load_dotenv
import os
import requests
import logging
from logging.handlers import RotatingFileHandler
import sys

# Configure logging to both file and console
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = RotatingFileHandler('tavily_client.log', maxBytes=1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.DEBUG)

# Root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class TavilyClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError('TAVILY_API_KEY environment variable is required')
        logger.debug(f"API Key loaded: {'*' * 4}{self.api_key[-4:]}")
        self.base_url = 'https://api.tavily.com/v1'

    def search_news(self, query, max_results=10, time_range="week"):
        """
        Search for news using Tavily API with optimized parameters for news content
        """
        try:
            logger.debug(f"Making search request for query: {query}")
            headers = {
                "content-type": "application/json",
                "api-key": self.api_key
            }
            payload = {
                "query": query,
                "search_depth": "advanced",
                "topic": "news",
                "time_range": time_range,
                "max_results": max_results
            }
            logger.debug(f"Request URL: {self.base_url}/search")
            logger.debug(f"Request payload: {payload}")
            
            response = requests.post(
                f"{self.base_url}/search",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"API Error: Status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
            data = response.json()
            logger.debug(f"Received {len(data.get('results', []))} results")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None

    def extract_content(self, urls, include_images=False):
        """
        Extract content from specific URLs using Tavily API
        """
        try:
            logger.debug(f"Making extract request for URLs: {urls}")
            headers = {
                "content-type": "application/json",
                "api-key": self.api_key
            }
            payload = {
                "urls": urls,
                "include_images": include_images
            }
            
            response = requests.post(
                f"{self.base_url}/extract",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"API Error: Status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None