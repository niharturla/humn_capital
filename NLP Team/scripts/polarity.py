# Polarity Analysis
# Author: Kevin Huang

# pip install nltk
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json

sentiment = SentimentIntensityAnalyzer()
path = "example_segmented_transcript.txt"

data = open(path, 'r').read()

data = data.split('\n')
for i in range(len(data)):
    data[i] = data[i].split(' - ')
    data[i].append(sentiment.polarity_scores(data[i][1])['compound'])
    data[i].pop(0)

# write to JSON file
with open('example_segmented_transcript.json', 'w') as f:
    json.dump(data, f)