"""
YouTube Video Summarizer using LangChain and Together API

This script allows users to provide a YouTube video URL, extracts the transcript, and generates a concise 
summary along with the five most important points from the video content.

### Features:
- Validates if the provided URL is a valid YouTube link.
- Fetches the English transcript of the video.
- Uses the Together API with Meta-Llama-3.1-8B-Instruct-Turbo to summarize the transcript.
- Outputs a clear and engaging summary for the user.

### Usage:
1. Run the script via the command line. 
2. Enter a valid YouTube video URL when prompted.
3. The script fetches the transcript, processes it using the LLM, and outputs the summary.

### Author: Saasha Nair
"""

import time
import validators

from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api._errors import NoTranscriptFound

import api_keys

# Create an instance of the LLM to use
llm = ChatTogether(api_key=api_keys.TOGETHER_API_KEY,temperature=0.0, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

# Prompt to instruct the LLM to summarise the contents of the video transcript
prompt = PromptTemplate(
        input_variables=["video_transcript"],
        template="""
        Read through the entire transcript carefully.
            Provide a concise summary of the video's main topic and purpose.
            Extract and list the five most interesting or important points from the transcript. 
            For each point: State the key idea in a clear and concise manner.

            - Ensure your summary and key points capture the essence of the video without including unnecessary details.
            - Use clear, engaging language that is accessible to a general audience.
            - If the transcript includes any statistical data, expert opinions, or unique insights, 
            prioritize including these in your summary or key points.

        Video transcript: {video_transcript}    """
    )

def is_youtube_url(url):
    """Check if the provided url is valid. If yes, then ensure that it is from YouTube."""
    if not validators.url(url):
        return False
    url = url.split('://')[1].split('/')[0]
    return url == 'www.youtube.com'

def fetch_video_transcript(url):
    """Fetch the (english) transcript of a YouTube video using the YoutubeLoader."""
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    data = loader.load()
    return data[0].page_content

def summarise(video_transcript):
    """Generate a summary of the video using the transcript."""
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({
        "video_transcript": video_transcript,
    })
    return summary

if __name__ == "__main__":
    print('AI: Hello! Lovely to see you. Which video can I summarise for you?')
    url = input('Please enter the url: ') # accept the url from the user

    if is_youtube_url(url=url):
        video_trascript = fetch_video_transcript(url=url)

        print('\nHmm, give me a second, going through the video...')
        time.sleep(1.0) # added to give the illusion of thinking/processing

        print(f'\nHere is the summary:\n\n{summarise(video_transcript=video_trascript)}')
    else:
        print('Sorry. I can only work with Youtube videos.')