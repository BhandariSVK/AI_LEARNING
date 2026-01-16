#   tensorflow
#   fastapi
#   uvicorn
#   python-multipart
#   pillow
#   tensorflow-serving-api
#   matplotlib
#   numpy


from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()
MODEL = tf.saved_model.load("../createdModel/1")
# print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLl")
# print(MODEL.signatures)
INFER = MODEL.signatures["serving_default"]
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hellow ! From Server."


#! Prediction

@app.post("/predict")
async def predict(
    file:UploadFile = File(...)
):
    data = await file.read()
    img = Image.open(BytesIO(data))
    img = img.resize((256, 256))
    img = np.array(img)
    img_btch = np.expand_dims(img, 0)
    # prediction = MODEL(img_btch)
    prediction = INFER(keras_tensor = img_btch)
    OUTPUT = prediction["output_0"].numpy()
    prdted_class = CLASS_NAMES[int(np.argmax(OUTPUT[0]))]
    confidence = float(np.max(OUTPUT[0]))
    return { "prdted_class" : prdted_class, "confidence" : (confidence * 100) }



################ ENTRY POINT ##########################
if __name__=="__main__":
    uvicorn.run(app=app, host="localhost", port=8000)