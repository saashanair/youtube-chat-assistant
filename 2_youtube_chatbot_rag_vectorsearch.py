import validators

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_together import ChatTogether, TogetherEmbeddings

import api_keys
llm = ChatTogether(api_key=api_keys.TOGETHER_API_KEY,temperature=0.0, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

rag_prompt = PromptTemplate(
    input_variables=["context", "user_query"],
    template="""
        You are a helpful AI assistant. Read through the context carefully. Answer the user query based ONLY on the provided context.

        - Use clear, engaging language that is accessible to a general audience.
        - Always keep your answers concise and to the point.
        - Do mention any statistical data, expert opinions, or unique insights present in the transcript, where applicable.

        Context: {context}

        User Query: {user_query}

        If the answer is not in the context, simply say "I don't know." Do not make up an answer.
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
    

def chat_with_transcript(url):
    """Initiate the interactive chat session allowing the user to ask questions about the YouTube video."""
    video_transcript = fetch_video_transcript(url=url)
    vectorstore = create_vector_store(video_transcript=video_transcript)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    print('AI: Okay. The vectorstore is ready. What queries would you like answered today?')

    while True:
        user_query = input('You: ')

        if user_query.lower().strip() in ['exit', 'quit']:
            print('AI: Goodbye! Hope you found this chat helpful. :)')
            break

        # block of code to see the chunks retrieved from the vectorstore
        #rd = retriever.invoke(user_query)
        #for i, d in enumerate(rd):
        #    print(i, d.page_content)
        
        chain = (
            {
                "context": retriever | format_docs,
                "user_query": RunnablePassthrough(),
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(user_query)
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


