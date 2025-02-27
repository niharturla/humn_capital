# Formats segmented transcript as JSON (text, time, speaker, sentiment)
# Author: Kevin Huang

# pip install openai
# pip install nltk

# This uses the output from transcript.py

from openai import OpenAI
import json
segmented_transcript = open('segmented_transcript.txt', 'r').read()
cleaned_transcript = ""

# Sentiment Analysis
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Remove the index from the segmented transcript (we don't need it)
for line in segmented_transcript.split('\n'):
    data = line.split(' - ')
    cleaned_transcript += data[1] + ' - ' + data[2] + '\n'
cleaned_transcript = cleaned_transcript[:-1]

# Array of dictionaries to format the JSON
result = [{} for _ in range(len(cleaned_transcript.split('\n')))]

# Label the speakers and generate the corrected transcript
def generate_corrected_transcript(temperature, transcript) -> str:
    openai_key = 'fee-fi-fo-fum'
    openai_client = OpenAI(api_key=openai_key)
    system_prompt = "You are a helpful assistant whose task is to label transcribed text of a 1-on-1 interview between a manager an an employee. Please label the speakers, using the format 'Manager:' and 'Employee:' to indicate who is speaking. Attach this information to the end of each line using ' - ', using a 1 for manager and a 0 for employee."

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    )

    return response.choices[0].message.content

# `newer_transcript` has speaker labels.
# `final_transcript` will add sentiment analysis to each line.
newer_transcript = generate_corrected_transcript(0.5, cleaned_transcript)
final_transcript = ""
sentiment = SentimentIntensityAnalyzer()

for i, line in enumerate(newer_transcript.split('\n')):
    data = line.split(' - ')
    final_transcript += line + ' - ' + str(sentiment.polarity_scores(data[0])['compound']) + '\n'

final_transcript = final_transcript[:-1]

with open('newer_transcript.txt', 'w') as f:
    f.write(newer_transcript)

result = [{} for _ in range(len(newer_transcript.split('\n')))]

for i, line in enumerate(final_transcript.split('\n')):
    line = line.split(' - ')
    result[i] = {"text": line[0], "time": line[1], "speaker": int(line[2].split(':')[1]), "sentiment": float(line[3])}


# Convert to json with variables "text", "time", "speaker", and "sentiment"
with open('segmented_transcript.json', 'w') as f:
    json.dump(result, f)


