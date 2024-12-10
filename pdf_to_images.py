#! pip install PyPDF2
#! pip install pdf2image
#! apt-get install poppler-utils

from pdf2image import convert_from_path
import os

pdf_path = input('enter path of source pdf')
output_path=input('enter path of output folder')
os.makedirs(output_path)
start_page=input('enter start page')
stop_page=input('enter stop page')
# Convert PDF pages to images
images = convert_from_path(pdf_path,first_page=start_page, last_page=stop_page)
print('pages',len(images))
for it,page in enumerate(images):
    # Convert each page image to grayscale
    page = page.convert('L')
    #plt.imshow(page)
    page.save(f"{output_path+pdf_path.split[-1].replace('.pdf','')}page_{start_page + it}.png")
