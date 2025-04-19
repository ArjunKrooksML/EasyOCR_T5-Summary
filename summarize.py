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
        logging.warning("Input text for summarization is empty.")
        return "Input text is empty. Cannot generate summary."

    if summarizer_pipeline is None:
        logging.error("Summarization pipeline not loaded.")
        return "Error: Summarization model not available."

    logging.info(f"Generating summary (max_length={max_length}, min_length={min_length})...")

    try:
        summary_result = summarizer_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        generated_summary = summary_result[0]['summary_text']
        logging.info("Summary generated successfully.")
        return generated_summary
    except IndexError:
         logging.error("Summarization pipeline returned an unexpected empty result.")
         return "Error: Failed to generate summary (empty result)."
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        if "maximum sequence length" in str(e):
            return f"Error: Input text is too long for the model's limit ({max_input} tokens)."
        return f"Error during summarization: {e}"

if __name__ == '__main__':
    print("Done!")

