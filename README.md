# Invoice Q&A — LayoutLM Gradio Demo

A tiny web UI to ask questions about invoices (Colab-ready).
Upload an invoice image, type a question (e.g. “What is the due date?”), and the app returns the model’s extracted answer. Built with Hugging Face document-question-answering pipeline and a Gradio frontend.

### Table of contents

Description
Features
Tech stack
Quick Colab (one-click) setup
Local setup
Example usage
Output format
Troubleshooting
FAQs
License & credits

### Description

This project wraps the Hugging Face model impira/layoutlm-invoices in a Gradio web interface so you can interactively ask questions about invoices. Useful for quick testing, prototyping, and demos. Not production-ready — use a cloud invoice extraction API for scale.

Features

Upload image or use webcam

Ask natural-language questions about invoices

Returns best answer (and can be extended for scores / bounding boxes)

Runs in Colab or locally (with Tesseract installed)

Minimal, clean Gradio UI

## Tech stack

Python 3.8+

Hugging Face Transformers (pipeline — document-question-answering)

PyTorch (CPU or CUDA)

Tesseract OCR (system binary) + pytesseract

Pillow (PIL)

Gradio (web UI)

Optional: requests for remote images

## Quick Colab setup (paste into a Colab cell)
# 1) System & Python deps
!apt-get update -y
!apt-get install -y tesseract-ocr libtesseract-dev

# 2) Python packages
!pip install -q transformers torch torchvision accelerate pillow pytesseract gradio

# 3) Minimal demo code (run as a cell)
from transformers import pipeline
from PIL import Image
import gradio as gr

qa = pipeline("document-question-answering", model="impira/layoutlm-invoices")

def ask_invoice(image, question):
    try:
        img = Image.open(image) if isinstance(image, str) else image
        res = qa(image=img, question=question)  # returns list of answers
        return res[0]['answer'] if len(res)>0 else "No answer found"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## Invoice Q&A — LayoutLM Demo")
    img = gr.Image(type="pil", label="Invoice")
    q = gr.Textbox(placeholder="e.g. What is the invoice number?", label="Question")
    out = gr.Textbox(label="Answer")
    btn = gr.Button("Ask")
    btn.click(ask_invoice, inputs=[img, q], outputs=out)

demo.launch(share=True)


## Notes:

share=True gives a public URL in Colab.

Colab GPUs are optional; CPU works but may be slow for large models.

Local setup (Ubuntu example)
# 1. Install system Tesseract
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev

# 2. Create venv and install Python deps
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install transformers torch torchvision accelerate pillow pytesseract gradio

# 3. Save the Gradio demo as app.py and run:
python app.py


Windows / Mac: install Tesseract via installer or brew install tesseract (macOS). Ensure tesseract is in PATH.

Example app.py (full, ready-to-run)
from transformers import pipeline
from PIL import Image
import gradio as gr

qa = pipeline("document-question-answering", model="impira/layoutlm-invoices")

def ask_invoice(image, question):
    # image arrives as PIL Image (Gradio uses PIL when type="pil")
    try:
        result = qa(image=image, question=question)
        if result and len(result)>0:
            # return best answer with confidence
            a = result[0]
            return f"{a.get('answer','')}  (score: {a.get('score', None)})"
        return "No answer found"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Invoice Q&A — LayoutLM (impira/layoutlm-invoices)")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload invoice")
        with gr.Column():
            question = gr.Textbox(placeholder="What is the invoice number?", label="Question")
            answer = gr.Textbox(label="Answer")
            btn = gr.Button("🔎 Ask")
            btn.click(ask_invoice, inputs=[inp, question], outputs=answer)

demo.launch()

## How to use

Start the app (Colab or local).

Upload invoice image (PNG/JPG).

Type question in plain English (e.g., “what is the due date?”).

Click Ask.

Read the returned answer (and score if enabled).

Output format

The HF pipeline returns a list of dicts like:

[
  {
    "answer": "2025-09-03",
    "score": 0.98,
    "start": 125,
    "end": 135,
    "id": "0"
  }
]


Your UI can present answer and optionally score. If no answers, the list will be empty.

### Troubleshooting / Common errors

ValueError: pytesseract not available
=> Install Tesseract system binary (apt-get install tesseract-ocr) and pip install pytesseract.

OCR produces garbage / no text
=> Low image quality, rotated pages, or tiny font. Preprocess: binarize, deskew, increase DPI.

Model download very slow / out of RAM
=> Use Colab GPU or smaller model. Caching will happen in ~/.cache/huggingface.

CUDA mismatch / torch error
=> Install the correct torch wheel for your CUDA version or use CPU-only pip install torch --index-url ... per PyTorch instructions.

Empty answers for short questions
=> The pipeline needs word boxes to succeed for some images; consider running OCR separately (PaddleOCR) and passing word_boxes if you want more control.

# FAQ (short & brutal)

Q: Is this production-ready?
A: No. This is a demo/prototype. For production use Google Document AI, Azure Form Recognizer, or AWS Textract.

Q: Why is the OCR incorrect?
A: Because Tesseract is basic. For better OCR accuracy, use PaddleOCR or a commercial OCR API.

Q: Can it process PDFs?
A: Convert PDF pages to images (pdf2image, pdfplumber) then feed images to the pipeline.

Q: Does it run offline?
A: Yes, once the model is downloaded and you have local Tesseract and PyTorch. The first run downloads ~hundreds of MB.

Q: How to get bounding boxes for the answer?
A: The HF pipeline output may include span positions — to show bounding boxes you must run OCR to get token boxes and map models tokens to coordinates. This requires extra code.

Q: Can I process batches?
A: You can loop through images; for performance, modify pipeline to batch inputs, or use model inference with optimized batching.

Q: Licensing?
A: The model impira/layoutlm-invoices is under a specific license (check model page). Your app/code can use any license (e.g., MIT), but respect the model’s license (likely CC and may restrict commercial use).

Q: Want better accuracy than LayoutLM?
A: Try DONUT (end-to-end), TrOCR+postprocessing, or commercial APIs.

##Next steps / Improvements

Add overlay that highlights the region where the answer comes from (requires token->bbox mapping).

Add batch upload & CSV export of extracted fields.

Replace Tesseract with PaddleOCR for better text detection.

Fine-tune a LayoutLM model on your invoice templates for higher accuracy.

### Contributing

PRs welcome. Open issues for bugs or feature requests (bounding boxes, batch export, PDF support).

### License & credits

Project license: MIT (or your choice).

Model license: See Hugging Face model page (impira/layoutlm-invoices) — respect its license (may be CC BY-NC-SA or similar).

Built with: Transformers, PyTorch, Gradio, Tesseract.
