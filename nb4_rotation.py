# Imports
import requests
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
# Display the image with matplotlib
import matplotlib.pyplot as plt
from doctr.utils.geometry import rotate_image
from doctr.utils.geometry import rotate_image

# Download a sample
# !wget https://github.com/mindee/doctr/releases/download/v0.1.0/back_cover.jpg

# img = plt.imread('back_cover.jpg')
# plt.imshow(img); plt.axis('off'); plt.show()

#
# V1
#

doc = DocumentFile.from_images(['back_cover.jpg'])

predictor = ocr_predictor(
    pretrained=True,
    det_arch="fast_base",
    reco_arch="parseq",
    assume_straight_pages=False,
    detect_orientation=True,
    # disable_crop_orientation=True,
    # disable_page_orientation=True,
    straighten_pages=False
)  # .cuda().half() uncomment this line if we run on GPU
result = predictor(doc)

# Visualize the result
# result.show()

# Export the result to json like dictionary
json_export = result.export()
print("#-------------------------------crop and page orien off")
print(f"Detected orientation: {json_export['pages'][0]['orientation']['value']} degrees")
print()
print(f"Extracted text:\n{result.render()}")

#
# V2
#

doc = DocumentFile.from_images(['back_cover.jpg'])
# Let's rotate the document by 180 degrees
doc = [rotate_image(doc[0], 180, expand=False)]

predictor = ocr_predictor(
    pretrained=True,
    det_arch="fast_base",
    reco_arch="parseq",
    assume_straight_pages=False,
    detect_orientation=True,
    # disable_crop_orientation=False,
    # disable_page_orientation=False,
    straighten_pages=False
)  # .cuda().half() uncomment this line if we run on GPU
result = predictor(doc)

# Visualize the result
# result.show()

# Export the result to json like dictionary
json_export = result.export()
print("#-------------------------------crop and page orien on")
print(f"Detected orientation: {json_export['pages'][0]['orientation']['value']} degrees")
print()
print(f"Extracted text:\n{result.render()}")

#
# V3
#

doc = DocumentFile.from_images(['back_cover.jpg'])
# Let's rotate the document by 180 degrees
doc = [rotate_image(doc[0], 180, expand=False)]

predictor = ocr_predictor(
    pretrained=True,
    det_arch="fast_base",
    reco_arch="parseq",
    assume_straight_pages=False,
    detect_orientation=True,
    # disable_crop_orientation=False,
    # disable_page_orientation=False,
    straighten_pages=True
)  # .cuda().half() uncomment this line if we run on GPU
result = predictor(doc)

# Visualize the result
# result.show()

# Export the result to json like dictionary
json_export = result.export()
print("#-------------------------------straighten_pages to True")
print(f"Detected orientation: {json_export['pages'][0]['orientation']['value']} degrees")
print()
print(f"Extracted text:\n{result.render()}")