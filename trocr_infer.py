from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
import requests
from PIL import Image
#
image_path='path to image file'
image=Image.open(image_path).convert("RGB")
#
model_name=input('enter the model name e.g. mohammadalihumayun/trocr-ur')
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
#
pixel_values = processor(image, return_tensors="pt").pixel_values#.to('cuda')
print(pixel_values.shape)
# generating text
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
image
