import sys
import os

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Environment variables:")
for key, value in os.environ.items():
    if 'KEY' in key.upper():  # Only show API key related vars, masked
        print(f"{key}: {'*' * 8}")
    
try:
    import requests
    print("\nRequests package version:", requests.__version__)
except ImportError:
    print("\nRequests package not installed")

try:
    from dotenv import load_dotenv
    print("python-dotenv package installed")
    load_dotenv()
    tavily_key = os.getenv('TAVILY_API_KEY')
    print("TAVILY_API_KEY present:", bool(tavily_key))
except ImportError:
    print("python-dotenv package not installed")