###########################################
# Goals
# - Run models
# - Test runtimes
# - Test accuracy
###########################################

import timeit # returns time in seconds

# for full script runtime
start_time_full_script = timeit.default_timer()

# Imports
print("-------------------------------------------------")
print("Run imports...")
start_time_imports = timeit.default_timer()
###
from onnx.onnx_cpp2py_export import ONNX_ML
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
###
elapsed = timeit.default_timer() - start_time_imports
print("runtime to run imports: ", elapsed)



# load model
print("-------------------------------------------------")
print("Load model...")
start_time_load_model = timeit.default_timer()
###
model = ocr_predictor('fast_base','crnn_mobilenet_v3_small', pretrained=True)
###
elapsed = timeit.default_timer() - start_time_load_model
print("runtime to load model: ", elapsed)

# load pdf
print("-------------------------------------------------")
print("Load doc ...")
doc = DocumentFile.from_images("test_files/Handwriting-sample-from-for-NIST-SD19-a-Handwritten-sample-form-b-Images-of.png")


# Analyze
print("-------------------------------------------------")
print("Run OCR...")
start_time_run_model = timeit.default_timer()
###
result = model(doc)
###
elapsed = timeit.default_timer() - start_time_run_model
print("runtime to run model: ", elapsed)

# https://github.com/mindee/notebooks/blob/main/doctr/quicktour.ipynb
print("-------------------------------------------------")
print("Print result...")
string_result = result.render()
print(string_result)

# runtime
print("-------------------------------------------------")
elapsed = timeit.default_timer() - start_time_full_script
print("runtime of full script: ", elapsed)