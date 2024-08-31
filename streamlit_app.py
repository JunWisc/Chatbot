import streamlit as st
import json
import zipfile
import os
import openai  
from gtts import gTTS
import base64
from dotenv import load_dotenv

# .env file load
load_dotenv()

# Read an API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Data Loading
# Set up the data directory based on the current file location
zip_dir = os.path.join(os.path.dirname(__file__), 'corpus.zip')  # zip file Location

# Examine directory validation
if not os.path.exists(zip_dir):
    raise FileNotFoundError(f"Cannot find the designated path: {zip_dir}")

# Load every data from zip file
def load_all_data(zip_dir):
    all_data = []
    extracted_dir_base = os.path.join(os.path.dirname(__file__), 'extracted_data')

    with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir_base)

    json_files = [f for f in os.listdir(extracted_dir_base) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(extracted_dir_base, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

    return all_data

# Function to extract and transform counseling data into the required format
def extract_data(json_data):
    extracted_data = []
    for counseling_record in json_data:
        if isinstance(counseling_record, dict):
            profile = counseling_record.get("Profile", {})
            conversations = counseling_record.get("Conversation", {})

            counseling_info = {
                "persona-ids": profile.get("persona-id", ""),
                "personas": profile.get("persona", ""),
                "emotions": profile.get("emotion", ""),
                "content": []
            }

            for conv_idx, conv_data in conversations.items():
                content = conv_data.get("Content", [])
                for contents in content:
                    counseling_info["content"].append({
                        "content": contents.get("HS01", ""),
                        "content": contents.get("SS01", ""),
                        "content": contents.get("HS02", ""),
                        "content": contents.get("SS02", ""),
                        "content": contents.get("HS03", ""),
                        "content": contents.get("SS03", "")
                    })

            extracted_data.append(counseling_info)

    return extracted_data

# Definition of Chatbot Class
class Chatbot:
    def __init__(self, counseling_data, model):
        self.counseling_data = counseling_data
        self.current_session = [{"role": "system", "content": "You are a helpful counseling assistant."}]
        self.initial_question = "Hello! I am an Chatbot."
        self.model = model  

    # Method to generate responses through OpenAI
    def get_openai_response(self, conversation):
        openai.api_key = API_KEY  
        response = openai.ChatCompletion.create(  
            model=self.model,  
            messages=conversation
        )
        return response.choices[0].message.content  # Return the generated response

    # Method to start a conversation
    def chat(self, user_input=None):
        if user_input:
            self.current_session.append({"role": "user", "content": user_input})
        conversation = self.current_session
        response = self.get_openai_response(conversation)
        self.current_session.append({"role": "assistant", "content": response})
        return response  # Return the generated response

# Load all data from the zip file
all_data = load_all_data(zip_dir)
# Extract data into the required format
extracted_data = extract_data(all_data)

# Streamlit app
st.title('Chatbot')
st.markdown("")
st.caption("A streamlit chatbot powered by OpenAI with Junhae")

with st.expander("About Chatbot", expanded=False):
    st.write(
        """
        - This chat program is created using Kaggle's corpus data.
        - Selection of GPT models is available ([default] gpt-3.5-turbo, gpt-4).
        - Please note that it will not work without entering the Chat GPT API key.
        """)
    st.markdown(" --- ")
    st.write(
        """
        - This program was created by Junhae Lee from University of Wisconsin-Madison.
        """)

with st.sidebar:
    st.markdown("""<style>div[class*="stTextInput"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 20px; font-weight: bold;}</style>""", unsafe_allow_html=True)
    st.markdown(" --- ")

    # Create radio buttons to select GPT model
    model = st.radio(label="GPT Model", options=["gpt-3.5-turbo", "gpt-4"])  # Provide options for GPT-3.5 and GPT-4 models
    # Make the radio button labels bold
    st.markdown("""<style>div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] 
                > p {font-size: 20px; font-weight: bold;}</style>""", unsafe_allow_html=True)

    st.markdown(" --- ")

    # Create radio buttons to select TTS voice
    tts_voice = st.radio(label="Select TTS Voice", options=["TTS On", "TTS Off"])
    st.markdown("""<style>div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {font-size: 20px; font-weight: bold;}</style>""", unsafe_allow_html=True)

    st.markdown(" --- ")

    if st.button(label="🔄️ Reset"):
        # Reset code
        st.session_state["chat"] = []
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm an Chatbot."}]
        st.session_state["check_reset"] = True

# Create chatbot instance
chatbot = Chatbot(extracted_data, model) 

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    # Direct use of initial_question property
    st.session_state["messages"].append({"role": "assistant", "content": chatbot.initial_question})

# Display current conversation
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if response := st.chat_input("Enter Text"):
    st.session_state["messages"].append({"role": "user", "content": response})
    st.chat_message("user").write(response)
    response = chatbot.chat(user_input=response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    # Convert assistant response to TTS
    if tts_voice != "TTS Off":
        if tts_voice == "TTS On":
            lang = 'en'
            tts = gTTS(text=response, lang=lang, slow=False)  # Text To Speech
            tts.save("response.mp3")  # Save coverted speech to a file

            # Play TTS audio
            audio_file = open("response.mp3", "rb")
            audio_bytes = audio_file.read()  # Audio_bytes variable
            # Encode the audio file to Base64 format
            audio_base64 = base64.b64encode(audio_bytes).decode()
            # HTML audio tag
            audio_html = f"""
            <audio autoplay style="display:none">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            # Insert the generated HTML into the Streamlit app to automatically play
            st.markdown(audio_html, unsafe_allow_html=True)