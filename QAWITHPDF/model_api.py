## here in this .py file i will be writting code for generating or loading model into my local system
## google gemini model with the help of google gemini api key
import os
import sys
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
from IPython.display import Markdown,display
from exception import customexception
from logger import logging

load_dotenv()


google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = google_api_key)


def load_model():

    try:
        model = Gemini(models = "gemini-pro",api_key = google_api_key)
        return model
    except Exception as e:
        raise customexception(e,sys)
       
