import streamlit as st
import time
from rag_chatbot import OptimizedRAGChatbot
import os

# Page configuration
st.set_page_config(
    page_title="Retail AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        padding-top: 0rem;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 30px;
    }
    
    .header h1 {
        color: #2c3e50;
        margin-bottom: 10px;
        font-size: 2.5rem;
    }
    
    .header p {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Message styling */
    .message {
        display: flex;
        margin-bottom: 20px;
        animation: fadeIn 0.5s ease-in;
    }
    
    .user-message {
        justify-content: flex-end;
    }
    
    .assistant-message {
        justify-content: flex-start;
    }
    
    .message-content {
        max-width: 70%;
        padding: 15px 20px;
        border-radius: 20px;
        font-size: 1rem;
        line-height: 1.5;
        word-wrap: break-word;
    }
    
    .user-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    
    .assistant-content {
        background: #f8f9fa;
        color: #2c3e50;
        border: 1px solid #e9ecef;
        margin-right: auto;
    }
    
    /* Avatar styling */
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 10px;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }
    
    /* Input area styling */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 20px;
        border-top: 1px solid #e0e0e0;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    .input-container {
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        gap: 10px;
        align-items: center;
    }
    
    /* Chat messages area */
    .chat-messages {
        padding-bottom: 120px;
        max-height: calc(100vh - 200px);
        overflow-y: auto;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
        margin-left: 60px;
    }
    
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4CAF50;
        animation: bounce 1.4s ease-in-out infinite both;
    }
    
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
        } 40% {
            transform: scale(1);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Status indicator */
    .status-indicator {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 10px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        z-index: 1000;
    }
    
    .status-ready {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-loading {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    /* Scrollbar styling */
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 40px 20px;
        color: #7f8c8d;
    }
    
    .welcome-message h3 {
        margin-bottom: 15px;
        color: #2c3e50;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .message-content {
            max-width: 85%;
        }
        
        .header h1 {
            font-size: 2rem;
        }
        
        .input-area {
            padding: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "chatbot_ready" not in st.session_state:
    st.session_state.chatbot_ready = False

# Initialize chatbot
@st.cache_resource
def initialize_chatbot():
    """Initialize the RAG chatbot"""
    try:
        dataset_path = r"C:\Users\paran\OneDrive\Desktop\Quantabase\working\dataset"
        if not os.path.exists(dataset_path):
            st.error(f"Dataset path not found: {dataset_path}")
            return None
        
        with st.spinner("🚀 Loading AI Assistant... This may take a moment"):
            chatbot = OptimizedRAGChatbot(dataset_path)
        return chatbot
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return None

# Load chatbot if not already loaded
if not st.session_state.chatbot_ready:
    st.session_state.chatbot = initialize_chatbot()
    if st.session_state.chatbot:
        st.session_state.chatbot_ready = True

# Header
st.markdown("""
<div class="header">
    <h1>🤖 Retail AI Assistant</h1>
    <p>Your intelligent companion for retail queries and support</p>
</div>
""", unsafe_allow_html=True)

# Status indicator
if st.session_state.chatbot_ready:
    st.markdown("""
    <div class="status-indicator status-ready">
        ✅ AI Assistant Ready
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-indicator status-loading">
        ⏳ Loading AI Assistant...
    </div>
    """, unsafe_allow_html=True)

# Chat messages container
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    # Welcome message if no conversation history
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-message">
            <h3>👋 Welcome to Retail AI Assistant!</h3>
            <p>I'm here to help you with retail queries, delivery information, policies, and more.</p>
            <p>Ask me anything about our services!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="message user-message">
                <div class="message-content user-content">
                    {message["content"]}
                </div>
                <div class="avatar user-avatar">
                    👤
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message assistant-message">
                <div class="avatar assistant-avatar">
                    🤖
                </div>
                <div class="message-content assistant-content">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input area
st.markdown('<div style="height: 120px;"></div>', unsafe_allow_html=True)  # Spacer for fixed input

# Create columns for input layout
col1, col2 = st.columns([6, 1])

with col1:
    # Text input
    user_input = st.text_input(
        "Message",
        placeholder="Ask me about retail services, delivery, policies...",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    # Send button
    send_clicked = st.button("Send", key="send_button", use_container_width=True)

# Handle message sending
if (send_clicked or user_input) and user_input.strip():
    if st.session_state.chatbot_ready and st.session_state.chatbot:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })
        
        # Show typing indicator
        with st.spinner("🤖 Thinking..."):
            # Get response from chatbot
            try:
                response = st.session_state.chatbot.chat(user_input)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except Exception as e:
                error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_response
                })
        
        # Clear the input and rerun to show new messages
        st.rerun()
        
    else:
        st.error("⚠️ AI Assistant is not ready yet. Please wait for initialization to complete.")

elif send_clicked and not user_input.strip():
    st.warning("💬 Please enter a message before sending.")

# Clear chat button in sidebar
with st.sidebar:
    st.markdown("### 🛠️ Chat Controls")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.conversation_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Ask about retail policies
    - Inquire about delivery services  
    - Get product information
    - Learn about return procedures
    - Request customer support
    """)
    
    st.markdown("---")
    st.markdown("### ⚡ Features")
    st.markdown("""
    - **Fast Responses**: Optimized for speed
    - **Context Aware**: Remembers conversation
    - **Accurate Info**: Based on your dataset
    - **User Friendly**: ChatGPT-like interface
    """)

# Auto-scroll to bottom (JavaScript)
if st.session_state.messages:
    st.markdown("""
    <script>
        const chatMessages = document.querySelector('.chat-messages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    </script>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; color: #7f8c8d; border-top: 1px solid #e0e0e0; margin-top: 40px;">
    <small>🤖 Powered by Retail AI Assistant | Built with Streamlit & RAG Technology</small>
</div>
""", unsafe_allow_html=True)
