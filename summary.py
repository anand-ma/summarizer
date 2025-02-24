from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import streamlit as st
import os
import re
import time

# To run locally:
# 1. Install the required packages:
#     pip install -qU -r requirements.txt
# 2. Run the Streamlit app:
#     streamlit run summary.py
# clear && python3.11 -m pip install -qU -r requirements.txt && streamlit run summary.py


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "YTUrlLoaded" not in st.session_state:
    st.session_state.YTUrlLoaded = None
if "video_qa_chain" not in st.session_state:
    st.session_state.video_qa_chain = None
if "yt_thumb" not in st.session_state:
    st.session_state.yt_thumb = None


all_languages = ['ab', 'aa', 'af', 'ak', 'sq', 'am', 'ar', 'hy', 'as', 'ay', 'az', 'bn', 'ba', 'eu', 'be', 'bho', 'bs', 'br', 'bg', 'my', 'ca', 'ceb', 'zh-Hans', 'zh-Hant', 'co', 'hr', 'cs', 'da', 'dv', 'nl', 'dz', 'en', 'eo', 'et', 'ee', 'fo', 'fj', 'fil', 'fi', 'fr', 'gaa', 'gl', 'lg', 'ka', 'de', 'el', 'gn', 'gu', 'ht', 'ha', 'haw', 'iw', 'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'iu', 'ga', 'it', 'ja', 'jv', 'kl', 'kn', 'kk', 'kha', 'km', 'rw', 'ko', 'kri', 'ku', 'ky', 'lo', 'la', 'lv', 'ln', 'lt', 'lua', 'luo', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt', 'gv', 'mi', 'mr', 'mn', 'mfe', 'ne', 'new', 'nso', 'no', 'ny', 'oc', 'or', 'om', 'os', 'pam', 'ps', 'fa', 'pl', 'pt', 'pt-PT', 'pa', 'qu', 'ro', 'rn', 'ru', 'sm', 'sg', 'sa', 'gd', 'sr', 'crs', 'sn', 'sd', 'si', 'sk', 'sl', 'so', 'st', 'es', 'su', 'sw', 'ss', 'sv', 'tg', 'ta', 'tt', 'te', 'th', 'bo', 'ti', 'to', 'ts', 'tn', 'tum', 'tr', 'tk', 'uk', 'ur', 'ug', 'uz', 've', 'vi', 'war', 'cy', 'fy', 'wo', 'xh', 'yi', 'yo', 'zu']
indian_languages = ['as', 'bn', 'bho', 'gu', 'hi', 'kn', 'ks', 'ml', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'si', 'ta', 'te', 'ur']

llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o-mini')

llama_model = ChatOpenAI(model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    openai_api_key = st.secrets["TOGETHER_API_KEY"],
                    openai_api_base = "https://api.together.xyz/v1")

mixtral_model = ChatOpenAI(model = "mistralai/Mixtral-8x22B-Instruct-v0.1",
                    openai_api_key = st.secrets["TOGETHER_API_KEY"],
                    openai_api_base = "https://api.together.xyz/v1")


def load_video_transcript(video_url):
    loader = YoutubeLoader.from_youtube_url(
        video_url,
        # language=['ta', 'en', 'te'],
        language = all_languages,
        translation='en',  # Translate to English
        add_video_info=False
    )

    st.session_state.yt_thumb = f'https://img.youtube.com/vi/{loader.video_id}/hqdefault.jpg' # set thumbnail image
    # Loads youtube video transcript
    documents = loader.load()
    return documents


def create_qa_chain(vectorstore, model):
    template = """You are a helpful AI assistant that answers questions about passed context only.
    context: {context}
    chat history: {chat_history}
    User Query: {question} 
    Answer above question in {lang}
    {human_input}"""
    
    # for some reason, {context} variable has to be injected for the prompt to work, {chat_history} variable works directly from the session state
    prompt = PromptTemplate(input_variables=['context', 'chat_history', 'human_input', 'question', 'lang'], template=template)

    memory = ConversationBufferMemory(
        memory_key='chat_history', # key to store the chat history in the memory same as the session state key
        return_messages=True,
        input_key="human_input", # key to store the user input in the memory
    )

    video_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory = memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return video_qa_chain
  

def display_chat_history():
    for role, avatar, message, lang in st.session_state.chat_history:
        with st.chat_message(role, avatar=avatar):
            st.write(message)


def process_url(video_url, model):
    website_data = load_video_transcript(video_url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    video_splits = text_splitter.split_documents(website_data)
    video_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


    # Create a Chroma vector store
    # it failed miserably, when changing language or model so we are using FAISS instead
    # vectorstore = Chroma.from_documents(video_splits, video_embeddings, collection_name="video_transcript")
    # Create a FAISS vector store
    vectorstore = FAISS.from_documents(video_splits, video_embeddings)
    
    if model == "Llama":
        model = llama_model
    elif model == "Mistral":
        model = mixtral_model

    # Create conversation chain
    st.session_state.video_qa_chain = create_qa_chain(vectorstore, model)

    st.session_state.YTUrlLoaded = True

    return True


def initiate_processing():
    # Reset the chat history and the YTUrlLoaded flag
    st.session_state.YTUrlLoaded = False
    st.session_state.chat_history = [] 

    if st.session_state.video_url:
        try:
            with st.spinner("Processing youtube url..."):
                success = process_url(st.session_state.video_url, st.session_state.model)
                if success:
                    st.success("Processing complete!")
        except Exception as e:
            st.error(f"This Video may not have Transcript, Try a different Video \nError: {str(e)}")
            # st.error(f"This Video may not have Transcript, Try a different Video")



# Frond End Code
    
# Page configuration
st.set_page_config(page_title="Video Summarizer", page_icon="🎥")

# Sidebar for url
with st.sidebar:
    st.subheader("Your YouTube URL")
    
    st.text_input(
        label="Youtube Url",
        type="default",
        # value="https://www.youtube.com/watch?v=5bqBre9wOLA",
        placeholder="Enter any YouTube video url",
        disabled=False,
        key="video_url", # store key in session state
        on_change=initiate_processing
    )

    st.selectbox(
        "Select AI Model",
        ("Mistral", "Llama"),
        key="model",
        on_change=initiate_processing
    )
    
st.header("Video Summary 📽")
st.subheader("Ask questions about the video or Summarize in any Language")

# Main chat interface
if st.session_state.YTUrlLoaded:

    user_question = st.chat_input("Ask a Question or Summarize",)
    
    lang = st.text_input(
        label="Language",
        value="English",
        max_chars=15,
        type="default",
        disabled=False
    )

    yt_thumbnail = st.session_state.yt_thumb
    st.image(yt_thumbnail, use_container_width=True)

    if user_question:
        user_role = "User"
        user_avatar = "👩‍🦰"

        # add question without waiting for answers
        st.session_state.chat_history.append((user_role, user_avatar, user_question, lang))

        # events like text input will rerun the app, hence the session state to preserve chat history, we are displaying 
        # the chat history and the last question asked by the user
        display_chat_history() 

        try:
            with st.spinner("Thinking..."):
                response = st.session_state.video_qa_chain({
                    "human_input":'',
                    "question": user_question,
                    "lang": lang
                })
                assistant_role = "Teacher"
                assistant_avatar = "👩‍🏫"
                st.session_state.chat_history.append((assistant_role, assistant_avatar, response["answer"], lang))

                # we are writing message directly instead of calling display_chat_history() to 
                # avoid displaying the last question twice
                with st.chat_message(assistant_role, avatar=assistant_avatar):
                    st.write(response["answer"])
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

# Display initial instructions
else:
    st.write("👈 Enter any YouTube url in the sidebar to get started!")
