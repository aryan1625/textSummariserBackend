
# from transformers import BartTokenizer, BartForConditionalGeneration
# import torch
import os
import requests
class TextSummarizer:
    def __init__(self):
        print("Loading model and tokenizer...")
        self.api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        token = os.getenv('token1', 'your_token_here')  
        self.headers = {
            "Authorization": f"Bearer {token}"
        }

    def summarize(self, text,mode = 'medium'):
        if not text.strip():
            return "Input text is empty. Please provide some content."
        print("Text received:", text[:100])  # Print part of the input
        print("Mode:", mode)
        
        #set mode
        mode = mode.lower()
        if(mode=='short'):
            max_length,min_length = 200,100
        elif(mode=='medium'):
            max_length,min_length = 300,150
        else:
            max_length,min_length = 400,200

        payload = {
            "inputs": text,
            "parameters": {
                "max_length":max_length,
                "min_length": min_length,
                "do_sample": False
            }
        }
        print("Sending request to HuggingFace API...")
        response = requests.post(self.api_url,headers = self.headers,json = payload)
        print("Response status code:", response.status_code)
        print("Response text:", response.text)

        if response.status_code == 200:
            try:
                return response.json()[0]["summary_text"]
            except (KeyError, IndexError):
                return "Failed to extract summary from response."
        elif response.status_code == 503:
            return "Model is loading on HuggingFace. Please try again after a few seconds."
        else:
            return f"API Error: {response.status_code} - {response.text}"

        