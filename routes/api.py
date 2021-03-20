from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from controller.model import Model, get_model

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    

@app.post("/detect_emotion_full", response_model=SentimentResponse)
def detect_emotion_full(request: SentimentRequest, model: Model = Depends(get_model)):
    sentiment = model.detect_emotion_full(request.text)
    return SentimentResponse(
        sentiment=sentiment
    )

@app.post("/detect_emotion_binary", response_model=SentimentResponse)
def detect_emotion_binary(request: SentimentRequest, model: Model = Depends(get_model)):
    sentiment = model.detect_emotion_binary(request.text)
    return SentimentResponse(
        sentiment=sentiment
    )