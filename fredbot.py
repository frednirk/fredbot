# Fred Chat bot created May 2024 Author Tony Duffy

# Python libraries
import streamlit as st
import os
from groq import Groq
import random

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


# Load environment variable file
load_dotenv()

# Load GROQ API Key
groq_api_key = st.secrets['GROQ_API_KEY']

# the main program control
def main():

    st.title("Fred Chat Bot")
    
    # Set bot memory to 5 interactions
    memory=ConversationBufferWindowMemory(value = 5)

    user_question = st.text_area("Ask Fred a question:")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input':message['human']},{'output':message['AI']})


    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name= "llama3-8b-8192"
    )

    conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
    )
    # Conversation interaction
    if user_question:
        response = conversation(user_question)
        message = {'human':user_question,'AI':response['response']}
        st.session_state.chat_history.append(message)
        st.write("Fred :  ", response['response'])

if __name__ == "__main__":
    main()
