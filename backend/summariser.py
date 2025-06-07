from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class TextSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        print("Loading model and tokenizer...")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        print("Model loaded successfully.")

    def summarize(self, text,mode = 'medium'):
        if not text.strip():
            return "Input text is empty. Please provide some content."
        
        #set mode
        mode = mode.lower()
        if(mode=='short'):
            max_length,min_length = 200,100
        elif(mode=='medium'):
            max_length,min_length = 300,150
        else:
            max_length,min_length = 400,200


        # Tokenize the text input
        inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)

        # Generate summary (you can play with parameters)
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

        # Decode and return the summary
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)