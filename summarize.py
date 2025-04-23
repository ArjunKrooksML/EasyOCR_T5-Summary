from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
"""
max_input = 1024
max_summ = 200
min_summ = 30
"""

def summarize_text(text: str, summarizer_pipeline,max_input=1024, max_length=200, min_length=40):
    if not text:
        
        return "Input text is empty. Cannot generate summary."

    if summarizer_pipeline is None:
        
        return "Error: Summarization model not available."

    

    try:
        summary_result = summarizer_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        generated_summary = summary_result[0]['summary_text']
        
        return generated_summary
    except IndexError:
         
         return "Error: Failed to generate summary (empty result)."
    except Exception as e:
        
        if "maximum sequence length" in str(e):
            return f"Error: Input text is too long for the model's limit ({max_input} tokens)."
        return f"Error during summarization: {e}"

if __name__ == '__main__':
    print("Done!")

