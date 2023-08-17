import os

import replicate
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

load_dotenv()

OpenAIAPI_Key = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ElevenLabs = os.getenv("ElevenLabs_API_KEY")

llm = OpenAI(temperature=.5)

def generate_haiku(mood): # function that takes in the mood and generates a haiku 
    prompt = PromptTemplate(
        input_variables=["mood"],
        template="You are a fantastic haiku writer, write me a {mood} haiku."
)
    chain = LLMChain(llm=llm, prompt=prompt)
    haiku = chain.run({
    'mood': mood,
        })
    return haiku


def generate_images(mood):
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": mood}
    )
    return output

def app():
    st.title("Lidi's Haiku Generator💚")
     
    with st.form(key='my_form'):
        mood = st.text_input(label="In one word describe how you feel right now")
        submit_button = st.form_submit_button("Generate Haiku")
        
        
        haiku = ""
        if submit_button:
            haiku = generate_haiku(mood) 
    
        st.markdown(haiku)


        images = generate_images(haiku)
        st.image(images[0])



if __name__ == "__main__":
    app()