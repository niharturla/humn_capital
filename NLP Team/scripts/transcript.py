# Transcript Generation
# Author: Kevin Huang

# pip install openai
# pip install moviepy

from openai import OpenAI
from moviepy.editor import VideoFileClip
import tempfile
import os

# Raw API keys
openai_key = 'fee-fi-fo-fum'

# Misc. variables
system_prompt = "You are a helpful assistant whose task is to correct any spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."
system_prompt_segments = "You are a helpful assistant whose task is to correct any spelling discrepancies in the transcribed text which is split into segments with format: 'id - text - average time'. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. Additionally, please throw out any segments that are not spoken words, such as 'um' or 'uh', or combine them with the surrounding segments only if it makes sense to do so."

video_path = 'segmented_video.mp4'
# For this 277 second video, when converted to an mp3 file, the size is 4433546 bytes
# The max size for the OpenAI API is 26214400 bytes
# So, an estimate of the max length of the video is 277 * 26214400 / 4433546 = 1632 seconds ~ 27 minutes
# If the video is longer than 27 minutes, split it into multiple segments using a package like pydub

# Initialize client
openai_client = OpenAI(api_key=openai_key)


def main():
    transcript_verbose = generate_verbose_transcript(video_path)

    # Generate transcript with segments
    segmented_transcript = ""
    for segment in transcript_verbose.segments:
        avg_time = (segment['start'] + segment['end']) / 2
        line = str(segment['id']) + ' - ' + str(segment['text']) + ' - ' + str(avg_time) + '\n'
        segmented_transcript += line

    # Generate transcript with identification
    corrected_segmented_transcript = generate_corrected_transcript_plaintext(0.5, system_prompt_segments, segmented_transcript)

    # Write to file
    with open('segmented_transcript.txt', 'w') as f:
        f.write(corrected_segmented_transcript)


def video_to_audio(video_path):
    """
    Convert video to audio
    :param video_path: path to video
    :return: path to audio file
    """
    video_clip = VideoFileClip(video_path)

    audio_data = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    video_clip.audio.write_audiofile(audio_data.name, codec='mp3')

    return audio_data.name


# Verbose transcript generation with timestamps
def generate_verbose_transcript(file_path):
    """
    Generate a verbose transcript from an audio file
    :param file_path: path to audio file
    :return: verbose transcript
    """
    audio_file = video_to_audio(file_path)

    with open(audio_file, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    os.remove(audio_file)

    return transcript


# Vanilla transcript generation
def generate_transcript(file_path):
    """
    Generate a transcript from an audio file
    :param file_path: path to audio file
    :return: transcript
    """

    audio_file = video_to_audio(file_path)

    with open(audio_file, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
        )

    os.remove(audio_file)

    return transcript


# Post-processing using GPT (use different system prompts for different tasks)
def generate_corrected_transcript(temperature, system_prompt, transcript) -> str:

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
                "content": transcript.text
            }
        ]
    )

    return response.choices[0].message.content


def generate_corrected_transcript_plaintext(temperature, system_prompt, transcript) -> str:

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


# Run the main function
if __name__ == "__main__":
    main()