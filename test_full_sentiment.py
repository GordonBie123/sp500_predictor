from news_scraper import NewsSentimentAnalyzer
import config

# Initialize analyzer
analyzer = NewsSentimentAnalyzer()

# Test fetching news for a stock
symbol = 'AAPL'
articles = analyzer.fetch_news(symbol, days_back=7)
print(f"Found {len(articles)} articles")

if articles:
    # Test analyzing single article
    first_article = articles[0]
    print(f"\nFirst article title: {first_article.get('title', 'No title')}")
    
    # Test sentiment on first article
    text = f"{first_article.get('title', '')} {first_article.get('description', '')}"
    sentiment = analyzer.analyze_sentiment(text)
    print(f"Sentiment: {sentiment}")
    
    # Test aggregated sentiment
    aggregated = analyzer.get_aggregated_sentiment(articles)
    print(f"\nAggregated sentiment: {aggregated}")