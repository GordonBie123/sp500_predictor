from newsapi import NewsApiClient

api_key = '93fea06edd8b41148e5c64949494225c'
newsapi = NewsApiClient(api_key=api_key)

try:
    # Test the API
    top_headlines = newsapi.get_top_headlines(q='stock', language='en', page_size=5)
    print(f"API Status: {top_headlines['status']}")
    print(f"Total Results: {top_headlines['totalResults']}")
except Exception as e:
    print(f"API Error: {e}")