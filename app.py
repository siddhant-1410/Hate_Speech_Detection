import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessing import preprocess_text
import time

# Page configuration
st.set_page_config(
    page_title="Hate Speech Detection Chatbot",
    page_icon="🛡️",
    layout="wide"
)

# Load model and preprocessing tools
@st.cache_resource
def load_artifacts():
    model = load_model("model/best_lstm_model.h5")
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("model/max_length.txt", "r") as f:
        max_length = int(f.read())
    return model, tokenizer, max_length

model, tokenizer, max_length = load_artifacts()
class_names = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm your Hate Speech Detection assistant. Send me any text and I'll analyze it for hate speech, offensive language, or classify it as neither. What would you like me to analyze?"
    })

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    .user-message {
        text-align: right;
        margin: 0.5rem 0;
    }
    
    .user-bubble {
        display: inline-block;
        background-color: #007bff;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 15px 15px 5px 15px;
        max-width: 70%;
        word-wrap: break-word;
    }
    
    .assistant-message {
        text-align: left;
        margin: 0.5rem 0;
    }
    
    .assistant-bubble {
        display: inline-block;
        background-color: #f1f3f4;
        color: #333;
        padding: 0.75rem 1rem;
        border-radius: 15px 15px 15px 5px;
        max-width: 70%;
        word-wrap: break-word;
    }
    
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .confidence-bar {
        background-color: rgba(255,255,255,0.3);
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 20px;
        background-color: #4CAF50;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🛡️ Hate Speech Detection Chatbot")
st.markdown("### Chat with AI to detect hate speech, offensive language, or neutral content")

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if "prediction" in message:
                # Format prediction response
                pred_class = message["prediction"]["class"]
                confidence = message["prediction"]["confidence"]
                
                # Get color based on prediction
                if pred_class == "Hate Speech":
                    color = "#ff4444"
                elif pred_class == "Offensive Language":
                    color = "#ff8800"
                else:
                    color = "#44ff44"
                
                confidence_width = int(confidence * 100)
                
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="prediction-box">
                        <strong>🤖 Analysis Result:</strong><br>
                        <div style="margin: 0.5rem 0;">
                            <strong>Classification:</strong> <span style="color: {color};">⚡ {pred_class}</span>
                        </div>
                        <div style="margin: 0.5rem 0;">
                            <strong>Confidence:</strong> {confidence:.1%}
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence_width}%;">
                                {confidence:.1%}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="assistant-bubble">
                        🤖 {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Input section
st.markdown("---")
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Type your message here...",
        placeholder="Enter text to analyze for hate speech...",
        key="user_input"
    )

with col2:
    send_button = st.button("Send 📤", use_container_width=True)

# Handle user input
if send_button and user_input.strip():
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    try:
        # Process the text
        with st.spinner("🔍 Analyzing text..."):
            processed = preprocess_text(user_input)
            seq = tokenizer.texts_to_sequences([processed])
            padded = pad_sequences(seq, maxlen=max_length)
            prediction = model.predict(padded)[0]
            pred_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            # Simulate processing time for better UX
            time.sleep(0.5)
        
        # Add assistant response with prediction
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Analysis complete!",
            "prediction": {
                "class": class_names[pred_class],
                "confidence": confidence
            }
        })
        
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Sorry, I encountered an error while analyzing your text: {str(e)}"
        })
    
    # Clear input and rerun to show new messages
    st.rerun()

elif send_button and not user_input.strip():
    st.warning("Please enter some text to analyze!")

# Sidebar with information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info("""
    **Classification Categories:**
    - 🔴 **Hate Speech**: Contains hateful content
    - 🟠 **Offensive Language**: Offensive but not hateful
    - 🟢 **Neither**: Clean, neutral content
    """)
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Be specific with your text
    - Try different types of content
    - Check the confidence score
    - Longer texts may be more accurate
    """)
    
    st.markdown("### 🔧 Actions")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Chat history cleared! What would you like me to analyze?"
        })
        st.rerun()
    
    # Show statistics
    if len(st.session_state.messages) > 1:
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Messages Analyzed", user_messages)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    🛡️ Hate Speech Detection Chatbot | Powered by Deep Learning
</div>
""", unsafe_allow_html=True)