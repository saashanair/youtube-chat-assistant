"""
YouTube Video Chat Assistant using LangChain and Together API

This script enables users to chat with a YouTube video. It allows the user to 
input a YouTube video URL, extracts the transcript and answers the users 
queries until the user exits the chat.

### Features:
- Validates if the provided URL is a valid YouTube link.
- Fetches the English transcript of the video.
- Uses the Together API with Meta-Llama-3.1-8B-Instruct-Turbo to summarize the transcript.
- Provides a command line based chat interface.
- Uses the entire video trascript as context to answer the users questions.

### Usage:
1. Run the script via the command line. 
2. Enter a valid YouTube video URL when prompted.
3. And then ask away! :)

### Author: Saasha Nair
"""

import validators

from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader

import api_keys

# Create an instance of the LLM to use
llm = ChatTogether(api_key=api_keys.TOGETHER_API_KEY,temperature=0.0, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

# Prompt to instruct the LLM to use the video transcript ONLY to answer the question asked
prompt = PromptTemplate(
            input_variables=["video_transcript", "user_query"],
            template="""
                Read through the entire transcript carefully. Respond to the user query based on only this transcript.

                - Ensure your response capture the essence of the video without including unnecessary details.
                - Use clear, engaging language that is accessible to a general audience.
                - Use appropriate emojis to make the response fun.
                - Always keep your answers concise and to the point.
                - Do mention any statistical data, expert opinions, or unique insights present in the transcript, where applicable.
                Do prioritise this is the query asks for a summary of the video.
                - Where possible, make the answers clear bulleted list. Use appropriate and applicable emoji as the bullets for the list.
                - If the query seems incomplete or the subject is not covered in the video transcript, let the user know that and then provide a list of 3 queries that you could help with relevant to the video. 

                Video transcript: {video_transcript}
                User Query: {user_query}"""
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

def chat_with_transcript(url):
    """Initiate the interactive chat session allowing the user to ask questions about the YouTube video."""
    video_transcript = fetch_video_transcript(url=url)

    while True:
        user_query = input('You: ')

        if user_query.lower().strip() in ['exit', 'quit']:
            print('AI: Goodbye! Hope you found this chat helpful. :)')
            break

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "video_transcript": video_transcript,
            "user_query": user_query,
        })
        print(f"AI: {response}")

if __name__ == "__main__":
    print('AI: Hello! Lovely to see you. Note: To end this conversation, please type \'exit\'.')
    print('AI: Which YouTube video would you like to gain insights from today?\n')
    
    while True:
        url = input('Please enter the url: ')
        if is_youtube_url(url=url):
            break
        else:
            print('Please ensure that you provide a YouTube url.')
    
    print('AI: Amazing! Let\'s go! Let me create the vectorstore for it.')
    chat_with_transcript(url=url)


