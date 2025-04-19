import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import easyocr
from ini_easyocr import extract_pdf 
from summarize import summarize_text


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@st.cache_resource
def load_ocr_reader(languages=['en']):

    logging.info(f"Cache miss: Loading easyocr Reader for languages: {languages}...")
    try:
        gpu_available = torch.cuda.is_available()
        logging.info(f"EasyOCR will use {'GPU' if gpu_available else 'CPU'}")
        # Use the imported easyocr library here, NOT ini_easyocr
        reader = easyocr.Reader(languages, gpu=gpu_available)
        logging.info("EasyOCR Reader loaded successfully.")
        return reader
    except Exception as e:
        logging.error(f"Failed to load easyocr Reader: {e}")
        st.error(f"Fatal Error: Failed to initialize EasyOCR. Check logs. Error: {e}")
        st.stop()


@st.cache_resource
def load_summarization_pipeline(model_name="t5-small"):
    """Loads the summarization pipeline using Streamlit's caching."""
    logging.info(f"Cache miss: Loading summarization model: {model_name}...")
    try:
        device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Summarizer using device: {'GPU' if device == 0 else 'CPU'}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer_pipeline = pipeline(
            "summarization", model=model, tokenizer=tokenizer, device=device
        )
        logging.info("Summarization model loaded successfully.")
        return summarizer_pipeline
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        st.error(f"Fatal Error: Failed to load summarization model ({model_name}). Check logs. Error: {e}")
        st.stop()


st.set_page_config(page_title="PDF OCR & Summarizer (EasyOCR + PyMuPDF)", layout="wide")
st.title("📄 PDF Summarizer using EasyOCR and T5")
st.markdown("""
Upload a PDF file. The system will extract text using **EasyOCR**
and then generate a summary using Google's T5 model
""")


st.sidebar.header("Options")

selected_langs = ['en']

model_choice = "t5-small"
summary_min_length = st.sidebar.slider("Minimum Summary Length", 30, 200, 50)
summary_max_length = st.sidebar.slider("Maximum Summary Length", 100, 500, 150)

if summary_max_length < summary_min_length:
    st.sidebar.warning("Max length cannot be less than min length. Adjusting max length.")
    summary_max_length = summary_min_length


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    st.info(f"Processing `{file_name}`...")


    try:
        ocr_reader = load_ocr_reader(selected_langs)
        summarizer_pipeline = load_summarization_pipeline(model_choice)
        if ocr_reader is None or summarizer_pipeline is None:

            pass
    except Exception as load_err:
         st.error(f"An unexpected error occurred loading models: {load_err}")
         st.stop()



    extracted_text = ""

    if 'ocr_reader' in locals() and ocr_reader is not None:
        with st.spinner("Extracting text from PDF using EasyOCR... Please wait."):
            try:
                # Call extract_pdf (imported from ocr_utils.py)
                extracted_text = extract_pdf(pdf_bytes, ocr_reader=ocr_reader)
                st.success("Text extracted successfully!")
            except RuntimeError as e:
                st.error(f"Extraction Error: {e}")
                st.warning("Text extraction failed. Please ensure the PDF file is valid.")

            except Exception as e:
                st.error(f"An unexpected error occurred during text extraction: {e}")
    else:
        st.error("OCR Reader failed to load. Cannot extract text.")



    if extracted_text and 'summarizer_pipeline' in locals() and summarizer_pipeline is not None:
        st.subheader("Extracted Text (First 1000 characters)")
        st.text_area("Extracted Content", extracted_text[:1000] + "...", height=200, disabled=True)

        st.subheader("Generated Summary")
        with st.spinner(f"Generating summary using {model_choice}... This might take a moment."):
            try:
                summary = summarize_text(
                    extracted_text,
                    summarizer_pipeline,
                    max_length=summary_max_length,
                    min_length=summary_min_length
                )
                st.success("Summary generated!")
                st.markdown(summary)
            except Exception as e:
                st.error(f"Summarization Error: {e}")
                if "out of memory" in str(e).lower():
                    st.warning("GPU out of memory during summarization.")
                elif "Input text is too long" in str(e):
                     st.warning(f"{e}")

    elif not extracted_text and 'summarizer_pipeline' in locals() and summarizer_pipeline is not None:
         st.warning("Could not generate a summary")
    elif not uploaded_file:
         st.info("Upload a PDF file.")

else:
    st.info("Upload a PDF file")

st.sidebar.markdown("---")
st.sidebar.markdown("A simple application for PDF summarization using EasyOCR!")