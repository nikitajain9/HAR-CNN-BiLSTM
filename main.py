from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf  

app = FastAPI(title="HAR CNN-BiLSTM API")

try:
    model = tf.keras.models.load_model("har_cnn_bilstm_model.keras")
except Exception as e:
    print(f"Error loading model: {e}")


# 3. Define the activity mapping (UCI HAR standard)
ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# 4. Define the input data structure
class SensorData(BaseModel):
    features: list  

@app.get("/")
def home():
    return {"status": "Online", "model": "UCI HAR Smartphone Classifier"}

@app.post("/predict")
def predict(input_data: SensorData):
    # Validate input length
    if len(input_data.features) != 561:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 561 features, got {len(input_data.features)}"
        )
    
    try:
        # Convert to numpy array and reshape for prediction
        data_array = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(data_array)
        label = int(prediction[0])
        
        return {
            "activity_id": label,
            "activity_name": ACTIVITY_MAP.get(label, "Unknown Activity")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))