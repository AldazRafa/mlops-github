import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).parent / "animal-cnn"
with (MODEL_DIR / "labels.json").open(encoding="utf-8") as file:
    ANIMALS = json.load(file)
with (MODEL_DIR / "preprocessing.json").open(encoding="utf-8") as file:
    PREPROCESSING = json.load(file)

IMAGE_SIZE = tuple(PREPROCESSING["image_size"])
COLOR_MODE = PREPROCESSING["color_mode"]
model = tf.keras.models.load_model(MODEL_DIR / "model.keras")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload/image")
async def upload_image(img: UploadFile = File(...)):
    with Image.open(img.file) as original_image:
        image = original_image.convert(COLOR_MODE).resize(IMAGE_SIZE)
        batch = np.expand_dims(np.asarray(image), axis=0)

    predictions = model.predict(batch, verbose=0)
    class_index = int(np.argmax(predictions, axis=1)[0])
    return ANIMALS[class_index]
