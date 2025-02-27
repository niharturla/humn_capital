# Question Segmentation
# Author: Kevin Huang

path = "example_transcript.txt"
example_transcript = open(path, 'r').read()

punctuation = ['.', '!', '?']

i = 0
while i < len(example_transcript):
    if example_transcript[i] in punctuation:
        example_transcript = example_transcript[:i+1] + '\n' + example_transcript[i+1:]
    i += 1


questions = []
for line in example_transcript.split('\n'):
    if '?' in line:
        questions.append(line[1:])

print('There are', len(questions), 'questions in the transcript:', '\n')
for q in questions:
    print(q)

print()

transcript_with_questions = open(path, 'r').read()
for q in questions:
    transcript_with_questions = transcript_with_questions.replace(q, '<|' + q + '|>')

print('Transcript with questions highlighted:', '\n', transcript_with_questions)