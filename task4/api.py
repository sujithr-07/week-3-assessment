
# ==========================================
# FASTAPI: DISTILBERT MODEL SERVING
# ==========================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import torch.nn.functional as F

# ------------------------------------------
# 1. Initialize FastAPI
# ------------------------------------------

app = FastAPI(
    title="DistilBERT Sentiment API",
    description="Sentiment prediction using fine-tuned DistilBERT",
    version="1.0"
)

# ------------------------------------------
# 2. Load model ONCE
# ------------------------------------------

MODEL_PATH = "./saved_distilbert"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("DistilBERT model loaded successfully!")

# ------------------------------------------
# 3. Request schema
# ------------------------------------------

class PredictionRequest(BaseModel):

    text: str


# ------------------------------------------
# 4. POST /predict endpoint
# ------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    # Check empty text
    if not request.text.strip():

        raise HTTPException(
            status_code=422,
            detail="Text cannot be empty"
        )

    # Tokenize input
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Model prediction
    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=1
        )

    # Get prediction
    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = predicted_class.item()

    label = model.config.id2label[predicted_class]

    confidence = confidence.item()

    return {
        "text": request.text,
        "label": label,
        "confidence": round(confidence, 4)
    }


# ------------------------------------------
# 5. Root endpoint
# ------------------------------------------

@app.get("/")
def home():

    return {
        "message": "DistilBERT Sentiment API is running"
    }
