###########################################
# Goals
# - Run models
# - Test runtimes
# - Test accuracy
###########################################
# Notes
# - Seems slower on GPU
# - - I suppose becasue it takes too much time to load model to GPU?
# - - Or maybe my GPU is too small?

import timeit # returns time in seconds

# for full script runtime
start_time_full_script = timeit.default_timer()

# Imports
print("-------------------------------------------------")
print("Run imports...")
start_time_imports = timeit.default_timer()
###

from PIL import Image
import pytesseract

###
elapsed = timeit.default_timer() - start_time_imports
print("runtime to run imports: ", elapsed)

# # load model
# print("-------------------------------------------------")
# print("Load model...")
# start_time_load_model = timeit.default_timer()
# ###
# model = ocr_predictor('fast_base','crnn_mobilenet_v3_small', pretrained=True).to(device)
# ###
# elapsed = timeit.default_timer() - start_time_load_model
# print("runtime to load model: ", elapsed)

# # load pdf
# print("-------------------------------------------------")
# print("Load doc ...")
# doc = DocumentFile.from_images("test_files/Handwriting-sample-from-for-NIST-SD19-a-Handwritten-sample-form-b-Images-of.png")

# Analyze
print("-------------------------------------------------")
print("Load model, Load image, Run OCR...")
start_time_run_model = timeit.default_timer()
###
result = pytesseract.image_to_string(Image.open('test_files/Handwriting-sample-from-for-NIST-SD19-a-Handwritten-sample-form-b-Images-of.png'))
###
elapsed = timeit.default_timer() - start_time_run_model
print("runtime to run model: ", elapsed)

# https://github.com/mindee/notebooks/blob/main/doctr/quicktour.ipynb
print("-------------------------------------------------")
print("Print result...")
# string_result = result.render()
print(result)

# Runtime
print("-------------------------------------------------")
elapsed = timeit.default_timer() - start_time_full_script
print("runtime of full script: ", elapsed)