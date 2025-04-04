import streamlit as st
from utils import write_message
from agent import generate_response

st.set_page_config("Ebert", page_icon=":movie_camera:")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"},
    ]

def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    with st.spinner('Thinking...'):
        response = generate_response(message)
        write_message('assistant', response)
        


for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

if prompt := st.chat_input("What is up?"):
    write_message('user', prompt)

    handle_submit(prompt)