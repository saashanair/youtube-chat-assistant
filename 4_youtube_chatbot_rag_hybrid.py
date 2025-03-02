import validators

from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_together import ChatTogether, TogetherEmbeddings

import api_keys
llm = ChatTogether(api_key=api_keys.TOGETHER_API_KEY, temperature=0.0, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

intent_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a helpful AI assistant that helps people chat with video transcripts. Based on the user query, that contains information about the recent chat with the user and their current question, classify the intent into one of the following categories:
    
    - "summary" - If the user is asking for a general summary, key points, main idea, or an overview. If the user wants to know what the video is about, the message it conveys or something along those lines, then that too falls under this category.
    - "fact" - If the user is asking for a specific fact, detail, or piece of information. This could be things like what does this mean, who is this person etc.
    - "opinion" - If the user is asking for a subjective take or interpretation. Questions like should i do this, what do you think about this, how should i handle this would fall under this category.

    Return just a single word -- summary, fact or opinion. Do not need to provide any explanation.

    User Query: {question}
    Intent:
    """
)

rag_prompt = PromptTemplate(
    input_variables=["context", "user_query"],
    template="""
        You are a helpful AI assistant. Read through the context carefully. Answer the user query based ONLY on the provided context.

        - Use clear, engaging language that is accessible to a general audience.
        - Always keep your answers concise and to the point.
        - Do mention any statistical data, expert opinions, or unique insights present in the transcript, where applicable.

        Context: {context}

        User Query: {user_query}
    """
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_vector_store(video_transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(video_transcript)
    embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=api_keys.TOGETHER_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

def detect_intent(query):
    """Uses LLM to classify query intent."""
    intent_chain = intent_prompt | llm | StrOutputParser()
    return intent_chain.invoke({"question": query}).strip().lower()

def hybrid_retriever(user_query, intent, video_transcript, vectorstore):

    if intent == 'summary':
        return [Document(page_content=video_transcript)]
    elif intent == 'fact':
        vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
        semantic_retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={'k': 5}), llm=llm)

        vector_docs = vector_retriever.invoke(user_query)
        semantic_docs = semantic_retriever.invoke(user_query)
        return semantic_docs + vector_docs

def get_chat_history(chat_history, history_length=3):
    recent_history = chat_history[-history_length:]
    return '\n'.join(recent_history)


def chat_with_transcript(url):
    """Initiate the interactive chat session allowing the user to ask questions about the YouTube video."""
    chat_history = []

    video_transcript = fetch_video_transcript(url=url)
    vectorstore = create_vector_store(video_transcript=video_transcript)

    print('AI: Okay. The vectorstore is ready. What queries would you like answered today?')

    while True:
        user_query = input('You: ')

        if user_query.lower().strip() in ['exit', 'quit']:
            print('AI: Goodbye! Hope you found this chat helpful. :)')
            break
        

        recent_history = get_chat_history(chat_history=chat_history, history_length=5)
        augmented_user_query = f'{recent_history}\nUser Query: {user_query}'

        intent = detect_intent(augmented_user_query)
        if intent in ['summary', 'fact']:
            chain = (
                {
                    "context": RunnableLambda(lambda q: hybrid_retriever(user_query=q, intent=intent, video_transcript=video_transcript, vectorstore=vectorstore)) | format_docs,
                    "user_query": RunnablePassthrough(),
                }
                | rag_prompt
                | llm
                | StrOutputParser()
            )
            response = chain.invoke(augmented_user_query)
        elif intent == 'opinion':
            response = 'Apologies, I have been instructed not to give my opinions. You could rephrase that question to see what the creator thinks about it.'
        else:
            response = 'Sorry, I did not quite catch that. Could you please rephrase your question?'
            
        chat_history.append(f'User: {user_query}')
        chat_history.append(f'AI: {response}')
        print(f'AI: {response}')
        


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


