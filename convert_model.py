from tensorflow import keras
import os

model_path = os.path.join("inference", "animal-cnn", "model.keras")
model = keras.models.load_model(model_path)

#model.save("unpacked_keras", zipped=False)

# …or push directly to your Hub repo (auto-creates it for you):
model.save("hf://iamrafa/mlops-2627")