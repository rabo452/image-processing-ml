import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from image_retrieval.search_image import getClosestImages

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the images directory
IMAGE_FOLDER = '/home/dimka/dataset'

# Route to serve files from the /images folder
@app.get("/images/{filename:path}")
async def get_image(filename: str):
    file_path = os.path.join(IMAGE_FOLDER, filename)
    file_path = file_path.replace('%20', ' ')
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

# Route to handle finding the closest images
@app.post("/images/find-closest")
async def get_closest(image: UploadFile = File(...)):
    # Ensure the uploaded file is an image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Read the uploaded image file
    image_file = Image.open(image.file).convert('RGB')
    return getClosestImages(image_file)