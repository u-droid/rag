import os
from openai import OpenAI
import base64
from PIL import Image

def convert_tiff_to_jpg(tiff_path):
    # Open the TIFF file
    jpg_path = tiff_path.replace('.tiff','.jpg')
    with Image.open(tiff_path) as img:
        # Convert the image to RGB (JPG does not support transparency)
        rgb_img = img.convert('RGB')
        # Save the image in JPG format
        rgb_img.save(jpg_path, 'JPEG')
    os.remove(tiff_path)
    return jpg_path


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def image_to_text(image_path):
    client = OpenAI()
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are OCR. What’s the text in this image?"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )
    return response.choices[0].message.content
