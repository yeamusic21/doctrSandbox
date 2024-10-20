from onnx.onnx_cpp2py_export import ONNX_ML
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
# PDF
doc = DocumentFile.from_images("Handwriting-sample-from-for-NIST-SD19-a-Handwritten-sample-form-b-Images-of.png")
# Analyze
result = model(doc)
# return result
# print(result)

# https://github.com/mindee/notebooks/blob/main/doctr/quicktour.ipynb
string_result = result.render()
print(string_result)