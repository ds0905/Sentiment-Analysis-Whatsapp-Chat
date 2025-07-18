import streamlit as st
from openai import OpenAI
import PyPDF2
import io
import re
import pandas as pd

st.set_page_config(
    page_title="Sentiment Analysis of WhatsApp chat (with AI/ML).",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant for WhatsApp chat analysis, powered by LLaMA 3 on Groq."}
    ]
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = "groq_api_key"  # Replace with your Groq API Key
if "file_confirmation" not in st.session_state:
    st.session_state.file_confirmation = ""
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None
if "uploaded_file_text" not in st.session_state:
    st.session_state.uploaded_file_text = None
if "whatsapp_df" not in st.session_state:
    st.session_state.whatsapp_df = None
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

def parse_whatsapp_to_df(file_text):
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?: [APMapm]{2})?) - (.*?): (.*)"
    rows = []
    for line in file_text.splitlines():
        match = re.match(pattern, line)
        if match:
            date, time_, sender, message = match.groups()
            rows.append({"date": date, "time": time_, "sender": sender, "message": message})
    if rows:
        df = pd.DataFrame(rows)
        return df
    return None

with st.sidebar:
    st.header("WhatsApp Chat Controls")
    if st.button("Clear Chat"):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant for WhatsApp chat analysis, powered by LLaMA 3 on Groq."}
        ]
        st.session_state.file_confirmation = ""
        st.session_state.last_uploaded = None
        st.session_state.uploaded_file_text = None
        st.session_state.whatsapp_df = None
        st.session_state.awaiting_response = False
        st.rerun()

    st.subheader("Upload WhatsApp Chat File")
    uploaded_file = st.file_uploader("Upload a .pdf or .txt WhatsApp chat export", type=["pdf", "txt"])

    if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded:
        file_text = ""
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                file_text += page.extract_text() or ""
        elif uploaded_file.type == "text/plain":
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_text = stringio.read()

        st.session_state.uploaded_file_text = file_text
        st.session_state.last_uploaded = uploaded_file

        # WhatsApp format detection
        wa_line_found = any(
            re.match(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}(?: [APMapm]{2})? - .+: .+", line)
            for line in file_text.splitlines()
        )
        if wa_line_found:
            st.session_state.file_confirmation = "WhatsApp chat uploaded! You can now analyze it."
            st.session_state.whatsapp_df = parse_whatsapp_to_df(file_text)
        else:
            st.session_state.file_confirmation = "File does not look like a WhatsApp chat export."
            st.session_state.whatsapp_df = None

    if st.session_state.file_confirmation:
        st.success(st.session_state.file_confirmation)

st.title("Sentiment Analysis of WhatsApp chat (with AI/ML).")

def render_chat():
    chat_to_display = st.session_state.chat_history
    if st.session_state.awaiting_response and chat_to_display[-1]["role"] == "user":
        for msg in chat_to_display[:-1]:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                        <div style='max-width: 70%; background-color: #d4edda; padding: 10px 14px; border-radius: 12px;'>
                            <div style='font-weight: bold; margin-bottom: 4px; color: #155724;'>You</div>
                            <div style='word-wrap: break-word;'>{msg['content']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                        <div style='max-width: 70%; background-color: #f1f0f0; padding: 10px 14px; border-radius: 12px;'>
                            <div style='font-weight: bold; margin-bottom: 4px; color: #333;'>Assistant</div>
                            <div style='word-wrap: break-word;'>{msg['content']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        # last user message
        msg = chat_to_display[-1]
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='max-width: 70%; background-color: #d4edda; padding: 10px 14px; border-radius: 12px;'>
                    <div style='font-weight: bold; margin-bottom: 4px; color: #155724;'>You</div>
                    <div style='word-wrap: break-word;'>{msg['content']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.info("Assistant is typing...")
    else:
        for msg in chat_to_display:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                        <div style='max-width: 70%; background-color: #d4edda; padding: 10px 14px; border-radius: 12px;'>
                            <div style='font-weight: bold; margin-bottom: 4px; color: #155724;'>You</div>
                            <div style='word-wrap: break-word;'>{msg['content']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                        <div style='max-width: 70%; background-color: #f1f0f0; padding: 10px 14px; border-radius: 12px;'>
                            <div style='font-weight: bold; margin-bottom: 4px; color: #333;'>Assistant</div>
                            <div style='word-wrap: break-word;'>{msg['content']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.write("### Chat")
render_chat()

user_input = st.chat_input("Type your WhatsApp analysis question here...")

# --- Chatbot main logic for WhatsApp analysis only ---
if user_input and not st.session_state.awaiting_response:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.awaiting_response = True
    st.rerun()

if (
    st.session_state.awaiting_response
    and st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user"
):
    # WhatsApp Chat Analysis
    if st.session_state.whatsapp_df is not None and st.session_state.uploaded_file_text:
        last_user_msg = st.session_state.chat_history[-1]["content"]
        lower = last_user_msg.lower()
        df = st.session_state.whatsapp_df

        # Simple analysis: top words
        if "top words" in lower or "most common words" in lower:
            all_msgs = " ".join(df["message"].values)
            words = re.findall(r'\b\w+\b', all_msgs.lower())
            stopwords = set([
                "the", "and", "to", "of", "in", "a", "is", "for", "on", "with", "at", "by", "an", "be", "this", "that", 
                "it", "are", "as", "from", "was", "but", "or", "so", "if", "i", "you", "we", "he", "she", "they", "me",
                "my", "your", "our", "us", "him", "her", "them", "just", "not"
            ])
            filtered = [w for w in words if w not in stopwords and len(w) > 2]
            freq = pd.Series(filtered).value_counts().head(10)
            st.markdown("#### Top 10 Most Common Words")
            st.write(freq)
            st.session_state.chat_history.append({"role": "assistant", "content": "Here are the top 10 most common words in this chat."})
            st.session_state.awaiting_response = False
            st.rerun()
        # Otherwise: pass to LLM with WhatsApp chat context
        else:
            with st.spinner("Assistant is typing..."):
                chat_history = st.session_state.chat_history.copy()
                chat_history.append({
                    "role": "system",
                    "content": f"The following WhatsApp chat has been uploaded. Use it as context for answering any user question, and only answer about this chat. Context:\n{st.session_state.uploaded_file_text[:15000]}"
                })
                if st.session_state.groq_api_key:
                    try:
                        client = OpenAI(
                            api_key=st.session_state.groq_api_key,
                            base_url="https://api.groq.com/openai/v1"
                        )
                        response = client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=chat_history,
                            stream=True
                        )
                        assistant_response = ""
                        for chunk in response:
                            content = chunk.choices[0].delta.content or ""
                            assistant_response += content
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    except Exception as e:
                        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "Please enter your Groq API Key in the sidebar to start chatting."})
            st.session_state.awaiting_response = False
            st.rerun()
    else:
        st.session_state.chat_history.append({"role": "assistant", "content": "Please upload a valid WhatsApp chat (.txt or .pdf) to start analysis."})
        st.session_state.awaiting_response = False
        st.rerun()