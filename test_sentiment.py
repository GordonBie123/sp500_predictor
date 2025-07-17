from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Test VADER
vader = SentimentIntensityAnalyzer()
test_text = "The stock market is doing great today!"

scores = vader.polarity_scores(test_text)
print("VADER Scores:", scores)
print("Compound score:", scores.get('compound', 'NOT FOUND'))

# Test with empty text
empty_scores = vader.polarity_scores("")
print("\nEmpty text scores:", empty_scores)