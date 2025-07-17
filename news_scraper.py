"""
News Scraper and Sentiment Analysis Module
Fetches news and analyzes sentiment for stock predictions
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
import config

class NewsSentimentAnalyzer:
    """
    Class to fetch news and analyze sentiment
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the news sentiment analyzer
        
        Args:
            api_key: NewsAPI key (if None, uses config)
        """
        self.api_key = api_key or config.NEWS_API_KEY
        self.newsapi = NewsApiClient(api_key=self.api_key)
        self.vader = SentimentIntensityAnalyzer()
        
    def fetch_news(self, symbol: str, company_name: str = None, 
                   days_back: int = 7) -> List[Dict]:
        """
        Fetch news articles for a given stock symbol
        
        Args:
            symbol: Stock ticker symbol
            company_name: Company name (optional, helps with search)
            days_back: Number of days to look back
            
        Returns:
            List of news articles with metadata
        """
        articles = []
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        # Build search query
        query = f"{symbol} stock"
        if company_name:
            query = f"{company_name} OR {symbol} stock"
        
        try:
            # Fetch articles from NewsAPI
            news_response = self.newsapi.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            if news_response['status'] == 'ok':
                for article in news_response['articles']:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'url': article.get('url', '')
                    })
                
                print(f"✅ Fetched {len(articles)} news articles for {symbol}")
            else:
                print(f"❌ Error fetching news: {news_response.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error with NewsAPI: {str(e)}")
            # Fallback to free sources if API fails
            articles.extend(self._fetch_fallback_news(symbol, company_name))
            
        return articles
    
    def _fetch_fallback_news(self, symbol: str, company_name: str = None) -> List[Dict]:
        """
        Fallback method to fetch news from free sources
        """
        articles = []
        
        # Yahoo Finance news (no API key required)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news[:20]:  # Limit to 20 most recent
                articles.append({
                    'title': item.get('title', ''),
                    'description': '',
                    'content': '',
                    'source': 'Yahoo Finance',
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'url': item.get('link', '')
                })
                
            print(f"✅ Fetched {len(articles)} articles from Yahoo Finance")
            
        except Exception as e:
            print(f"⚠️ Could not fetch Yahoo Finance news: {str(e)}")
            
        return articles
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1, 'polarity': 0}
        
        # Clean text
        text = self._clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except:
            polarity = 0
            subjectivity = 0
        
        return {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def get_aggregated_sentiment(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Get aggregated sentiment scores from articles
        """
        if not articles:
            return {
                'mean_compound': 0,
                'mean_positive': 0,
                'mean_negative': 0,
                'mean_polarity': 0,
                'mean_neutral': 0.5,
                'sentiment_std': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'article_count': 0
            }
        
        sentiments = []
        
        for article in articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
        
        # Convert to DataFrame for easy aggregation
        df = pd.DataFrame(sentiments)
        
        # Calculate aggregated metrics and convert to Python float
        aggregated = {
            'mean_compound': float(df['compound'].mean()),
            'mean_positive': float(df['positive'].mean()),
            'mean_negative': float(df['negative'].mean()),
            'mean_polarity': float(df['polarity'].mean()),
            'mean_neutral': float(df['neutral'].mean()) if 'neutral' in df else 0.5,
            'sentiment_std': float(df['compound'].std()) if len(df) > 1 else 0.0,
            'positive_ratio': float((df['compound'] > 0.1).sum() / len(df)),
            'negative_ratio': float((df['compound'] < -0.1).sum() / len(df)),
            'article_count': len(articles)
        }
        
        return aggregated
    
    def create_sentiment_features(self, symbol: str, company_name: str = None,
                            days_back: int = 7) -> pd.DataFrame:
        """
        Create sentiment features for ML models
        """
        print(f"DEBUG: Starting sentiment analysis for {symbol}")
        
        try:
            # Fetch news
            print("DEBUG: Fetching news...")
            articles = self.fetch_news(symbol, company_name, days_back)
            print(f"DEBUG: Found {len(articles)} articles")
            
            # Get aggregated sentiment
            print("DEBUG: Getting aggregated sentiment...")
            sentiment_features = self.get_aggregated_sentiment(articles)
            print(f"DEBUG: Sentiment features: {sentiment_features}")
            
            # Ensure all required keys exist
            required_keys = ['mean_compound', 'mean_positive', 'mean_negative', 
                            'mean_neutral', 'mean_polarity', 'sentiment_std',
                            'positive_ratio', 'negative_ratio', 'article_count']
            
            for key in required_keys:
                if key not in sentiment_features:
                    print(f"DEBUG: Missing key {key}, adding default")
                    sentiment_features[key] = 0.0 if key != 'mean_neutral' else 1.0
            
            # Add time-based features
            sentiment_features['sentiment_trend'] = 0
            sentiment_features['sentiment_volatility'] = 0
            
            print(f"DEBUG: Final sentiment features: {sentiment_features}")
            return pd.DataFrame([sentiment_features])
            
        except Exception as e:
            print(f"ERROR in create_sentiment_features: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return default sentiment features
            return pd.DataFrame([{
                'mean_compound': 0.0,
                'mean_positive': 0.0,
                'mean_negative': 0.0,
                'mean_neutral': 1.0,
                'mean_polarity': 0.0,
                'sentiment_std': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'article_count': 0,
                'sentiment_trend': 0,
                'sentiment_volatility': 0
            }])
    
    def _calculate_daily_sentiments(self, articles: List[Dict]) -> List[Dict]:
        """Calculate sentiment by day"""
        from collections import defaultdict
        
        daily_articles = defaultdict(list)
        
        for article in articles:
            try:
                date = pd.to_datetime(article['published_at']).date()
                daily_articles[date].append(article)
            except:
                continue
        
        daily_sentiments = []
        for date, day_articles in sorted(daily_articles.items()):
            sentiment = self.get_aggregated_sentiment(day_articles)
            sentiment['date'] = date
            daily_sentiments.append(sentiment)
            
        return daily_sentiments
    
    def _calculate_trend(self, daily_sentiments: List[Dict]) -> float:
        """Calculate sentiment trend (slope of linear regression)"""
        if len(daily_sentiments) < 2:
            return 0
        
        x = np.arange(len(daily_sentiments))
        y = [s['mean_compound'] for s in daily_sentiments]
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NewsSentimentAnalyzer()
    
    # Test with UNH
    symbol = 'UNH'
    company_name = 'UnitedHealth Group'
    
    # Fetch news and analyze sentiment
    sentiment_features = analyzer.create_sentiment_features(symbol, company_name)
    
    print("\nSentiment Features:")
    print(sentiment_features)
    print(f"\nFeature names: {sentiment_features.columns.tolist()}")