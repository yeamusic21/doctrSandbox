import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Download a sample
# !wget https://eforms.com/download/2019/01/Cash-Payment-Receipt-Template.pdf
# Read the file
doc = DocumentFile.from_pdf("Cash-Payment-Receipt-Template.pdf")
print(f"Number of pages: {len(doc)}")

# Instantiate a pretrained model
predictor = ocr_predictor(pretrained=True)

# Display the architecture
print(predictor)

# basic inference
result = predictor(doc)

# get visual 
result.show()

# JSON export
json_export = result.export()
print(json_export)

# XML export
# xml_output = result.export_as_xml()
# print(xml_output[0][0])

# Or if you only need the extracted plain text
string_result = result.render()
print(string_result)