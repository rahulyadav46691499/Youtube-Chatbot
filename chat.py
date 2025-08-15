import streamlit as st
import re
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Video Chatbot",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/70px-YouTube_full-color_icon_%282017%29.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: 20%;
    }
    
    .bot-message {
        background: #f8f9fa;
        color: #2c3e50;
        border-left: 4px solid #667eea;
        margin-right: 20%;
    }
    
    .video-info {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    
    .stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'multi_query_retriever' not in st.session_state:
    st.session_state.multi_query_retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_loaded' not in st.session_state:
    st.session_state.video_loaded = False
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None

@st.cache_resource
def initialize_models():
    """Initialize the embedding model and LLM"""
    # Get API key from environment variable
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if not google_api_key:
        st.error("❌ GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        google_api_key=google_api_key,
        model='models/embedding-001'
    )
    
    llm = ChatGoogleGenerativeAI(
        api_key=google_api_key,
        model='gemini-2.5-flash'
    )
    
    prompt_template = PromptTemplate(
        template='''
        You are a helpful assistant.
        Answer the user query based on given context.
        If solution is not present inside context, just say IDK
        {context}

        Question: {user_query}
        ''',
        input_variables=['context', 'user_query']
    )
    
    return embedding_model, llm, prompt_template

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    pattern = r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def load_video_transcript(video_id, embedding_model, llm):
    """Load and process YouTube video transcript - your existing code"""
    try:
        # Your existing transcript processing code
        ytt_api = YouTubeTranscriptApi()
        transcripts = ytt_api.fetch(video_id, languages=['en', 'hi'])
        
        complete_transcript = ''
        for obj in transcripts.snippets:
            complete_transcript = complete_transcript + obj.text + ' '
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text = text_splitter.split_text(complete_transcript)
        
        # Create vector store
        vector_db = Chroma.from_texts(texts=text, embedding=embedding_model)
        
        # Create multi-query retriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(
                search_type='mmr',
                search_kwargs={'k': 3, "lambda_multi": 1}
            ),
            llm=llm
        )
        
        return vector_db, multi_query_retriever, True, None
        
    except Exception as e:
        return None, None, False, str(e)

def process_question(user_input, multi_query_retriever, prompt_template, llm):
    """Process user question - your existing code"""
    try:
        # Your existing retrieval and processing code
        closest_vectors = multi_query_retriever.invoke(user_input)
        
        context = ''
        for obj in closest_vectors:
            context = context + obj.page_content + '\n'
        
        final_prompt = prompt_template.invoke({'context': context, 'user_query': user_input})
        
        response = llm.invoke(final_prompt)
        
        return response.content
        
    except Exception as e:
        return f"Error processing question: {str(e)}"

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/70px-YouTube_full-color_icon_%282017%29.svg.png" width="50" style="vertical-align: middle; margin-right: 15px;">YouTube Video Chatbot</h1>
        <p>Ask questions about any YouTube video using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize models
    embedding_model, llm, prompt_template = initialize_models()
    
    # Sidebar for video loading
    with st.sidebar:
        st.markdown("### <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/50px-YouTube_full-color_icon_%282017%29.svg.png' width='25' style='vertical-align: middle; margin-right: 8px;'>Load YouTube Video", unsafe_allow_html=True)
        
        # Video URL input
        video_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )
        
        # Load video button
        if st.button("🔄 Load Video", use_container_width=True):
            if video_url:
                video_id = extract_video_id(video_url)
                if video_id:
                    with st.spinner("Loading video transcript..."):
                        vector_db, multi_query_retriever, success, error = load_video_transcript(
                            video_id, embedding_model, llm
                        )
                        
                        if success:
                            st.session_state.vector_db = vector_db
                            st.session_state.multi_query_retriever = multi_query_retriever
                            st.session_state.video_loaded = True
                            st.session_state.current_video_id = video_id
                            st.session_state.chat_history = []  # Clear chat history
                            st.success("✅ Video loaded successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Error loading video: {error}")
                else:
                    st.error("❌ Invalid YouTube URL")
            else:
                st.error("❌ Please enter a YouTube URL")
        
        # Video status
        if st.session_state.video_loaded:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #10b981, #059669);
                color: white;
                padding: 1rem;
                border-radius: 12px;
                border: none;
                margin: 1rem 0;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            ">
                <strong>🎬 Video Status:</strong><br>
                ✅ Loaded and ready<br>
                🔍 Ask questions below!
            </div>
            """, unsafe_allow_html=True)
            
            # YouTube video preview
            st.markdown("### <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/50px-YouTube_full-color_icon_%282017%29.svg.png' width='25' style='vertical-align: middle; margin-right: 8px;'>Video Preview", unsafe_allow_html=True)
            if st.session_state.current_video_id:
                st.video(f"https://www.youtube.com/watch?v={st.session_state.current_video_id}")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("👆 Load a YouTube video to start chatting!")
    
    # Main chat interface
    st.markdown("### 💬 Chat with the Video")
    
    # Chat history
    if st.session_state.chat_history:
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong> {bot_msg}
            </div>
            """, unsafe_allow_html=True)
    else:
        if st.session_state.video_loaded:
            st.info("💭 Ask your first question about the video!")
        else:
            st.info("📺 Load a YouTube video first to start asking questions.")
    
    # Question input (this replaces your user_input = input())
    if st.session_state.video_loaded:
        # Create a form for better UX
        with st.form(key="question_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_input = st.text_input(
                    "Your Question",
                    placeholder="Ask anything about the video...",
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
                submit_button = st.form_submit_button("Send", use_container_width=True)
            
            # Process question when form is submitted
            if submit_button and user_input:
                if st.session_state.multi_query_retriever:
                    with st.spinner("🤔 Thinking..."):
                        # This is where your user_input gets processed!
                        bot_response = process_question(
                            user_input, 
                            st.session_state.multi_query_retriever, 
                            prompt_template, 
                            llm
                        )
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_input, bot_response))
                        
                        # Rerun to show new messages
                        st.rerun()
                else:
                    st.error("❌ Please load a video first!")
    else:
        # Show input box even when no video is loaded, but disable it
        with st.form(key="question_form_disabled", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.text_input(
                    "Your Question",
                    placeholder="Load a video first...",
                    disabled=True,
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.form_submit_button("Send", disabled=True, use_container_width=True)

if __name__ == "__main__":
    main()