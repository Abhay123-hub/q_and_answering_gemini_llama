## here in this .py file i will be writting code for generating or loading model into my local system
## google gemini model with the help of google gemini api key
import os
import sys
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
from IPython.display import Markdown, display
import traceback

# Load environment variables
load_dotenv()

# Retrieve the Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure generative AI API if the key exists
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    print("Google API key not found. Please set it in your environment variables.")

def load_model():
    """
    Loads the Gemini model with the provided Google API key.

    Returns:
    - model (Gemini): The loaded Gemini model if successful.
    - False: If model loading fails.
    """
    try:
        # Check if the API key is valid
        if not google_api_key:
            raise ValueError("API key is missing or invalid.")

        # Load the Gemini model
        model = Gemini(models="gemini-pro", api_key=google_api_key)
        print("Model loaded successfully.")
        return model

    except Exception as e:
        # Print the traceback for debugging and return False
        print("Failed to load the model. See error details below:")
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
