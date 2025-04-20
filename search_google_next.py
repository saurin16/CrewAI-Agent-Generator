import requests
import os
import sys
from dotenv import load_dotenv

def search_google_next():
    # Open a file for writing output
    with open('search_results.txt', 'w') as f:
        def log(msg):
            f.write(msg + '\n')
            f.flush()
            print(msg, flush=True)  # Also try printing to console
            sys.stdout.flush()  # Force flush stdout
        
        load_dotenv()
        api_key = os.getenv('TAVILY_API_KEY')
        if not api_key:
            log("Error: TAVILY_API_KEY not found in environment")
            return

        url = "https://api.tavily.com/v1/search"
        headers = {
            "content-type": "application/json",
            "api-key": api_key
        }
        payload = {
            "query": "Google Next 2024 conference announcements latest news",
            "search_depth": "advanced",
            "topic": "news",
            "time_range": "week",
            "max_results": 5
        }

        try:
            log("Sending request to Tavily API...")
            response = requests.post(url, json=payload, headers=headers)
            log(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if results:
                    log("\nLatest Google Next News:\n")
                    for idx, result in enumerate(results, 1):
                        log(f"{idx}. {result.get('title')}")
                        log(f"   Source: {result.get('source')}")
                        log(f"   URL: {result.get('url')}")
                        log(f"   Snippet: {result.get('snippet')}\n")
                else:
                    log("No results found in the response")
            else:
                log(f"Error: API request failed with status {response.status_code}")
                log(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            log(f"Network error: {str(e)}")
        except Exception as e:
            log(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    search_google_next()