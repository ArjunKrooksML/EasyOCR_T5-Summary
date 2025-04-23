import io
from PIL import Image
import numpy as np
import fitz
import logging
import easyocr



def extract_pdf(pdf_file_bytes: bytes, ocr_reader: easyocr.Reader) -> str:

    if not ocr_reader:
        raise ValueError("EasyOCR reader instance is required.")

    extracted_text = ""
    doc = None

    try:
        
        doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        num_pages = doc.page_count
        


        for i in range(num_pages):
            page_num = i + 1
            
            page = doc.load_page(i)

            dpi = 150
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            

            try:
                img_bytes = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes))
                
            except Exception as img_conv_err:
                
                continue

            np_image = np.array(pil_image)

            
            results = ocr_reader.readtext(np_image, detail=0, paragraph=True)

            page_text = "\n".join(results)
            extracted_text += page_text + "\n\n"
            

        
        return extracted_text.strip()

    except fitz.fitz.FitzError as fitz_err:
         
         raise RuntimeError(f"Failed to process PDF using PyMuPDF. Error: {fitz_err}")
    except Exception as e:
    
        raise RuntimeError(f"Failed to extract text from PDF. Error: {e}")
    finally:
        if doc:
            doc.close()
            


if __name__ == '__main__':
    print("Done!")
