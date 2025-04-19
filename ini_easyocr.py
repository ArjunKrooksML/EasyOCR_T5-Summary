import io
from PIL import Image
import numpy as np
import fitz
import logging
import easyocr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_pdf(pdf_file_bytes: bytes, ocr_reader: easyocr.Reader) -> str:

    if not ocr_reader:
        raise ValueError("EasyOCR reader instance is required.")

    extracted_text = ""
    doc = None

    try:
        logging.info("Opening PDF with PyMuPDF...")
        doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        num_pages = doc.page_count
        logging.info(f"{num_pages} page(s) found.")


        for i in range(num_pages):
            page_num = i + 1
            logging.info(f"Processing page {page_num}/{num_pages}...")
            page = doc.load_page(i)

            dpi = 150
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            logging.info(f"Page {page_num}: Rendering successful (dpi={dpi}).")

            try:
                img_bytes = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes))
                logging.info(f"Page {page_num}: Converted pixmap to PIL Image.")
            except Exception as img_conv_err:
                logging.error(f"Page {page_num}: Failed to convert pixmap to PIL Image: {img_conv_err}")
                continue

            np_image = np.array(pil_image)

            logging.info(f"Page {page_num}: Performing OCR...")
            results = ocr_reader.readtext(np_image, detail=0, paragraph=True)

            page_text = "\n".join(results)
            extracted_text += page_text + "\n\n"
            logging.info(f"Page {page_num}: OCR successful.")

        logging.info("Finished processing all pages.")
        return extracted_text.strip()

    except fitz.fitz.FitzError as fitz_err:
         logging.error(f"PyMuPDF (Fitz) error: {fitz_err}")
         raise RuntimeError(f"Failed to process PDF using PyMuPDF. Error: {fitz_err}")
    except Exception as e:
        logging.error(f"Error during PDF processing or OCR: {e}")
        raise RuntimeError(f"Failed to extract text from PDF. Error: {e}")
    finally:
        if doc:
            doc.close()
            logging.info("PDF document closed.")


if __name__ == '__main__':
    print("Done!")