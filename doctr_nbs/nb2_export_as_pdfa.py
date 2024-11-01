# Imports
# import os
# import base64
# import re
# from tempfile import TemporaryDirectory

# from PyPDF2 import PdfMerger, PdfReader
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, from_hub
# from PIL import Image
# from ocrmypdf.hocrtransform import HocrTransform

# Download a sample
# !wget https://github.com/mindee/doctr/releases/download/v0.1.0/Versicherungsbedingungen-08-2021.pdf
# Read the file
docs = DocumentFile.from_pdf("test_files/Versicherungsbedingungen-08-2021.pdf")
# The document contains german text let's use a multilingual fine tuned model from the Hugging Face hub.
reco_model = from_hub("Felix92/doctr-torch-parseq-multilingual-v1")
model = ocr_predictor(det_arch='fast_base', reco_arch=reco_model, pretrained=True)
# we will grab only the first two pages from the pdf for demonstration
result = model(docs[:2])
result.show()

# DONT REALLY CARE ABOUT PDF/A, SO STOPPING HERE