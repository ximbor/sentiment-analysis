import os
import uvicorn
import logging
from transformers import pipeline
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = os.getenv("MODEL_NAME", "ximbor/sentiment-monitor")
NETWORK_INTERFACE = os.getenv("NETWORK_IF", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Init logging:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sentiment-api")

classifier = None

# Request model
class SentimentRequest(BaseModel):
    text: str

# App initialization:
@asynccontextmanager
async def lifespan(app: FastAPI):
    # App load:
    logger.info("Application started...")

    yield

    # App unload:
    logger.info("Application is shutting down...")
    app.state.model_ready = False
    del classifier


def load_model():
    try:
        logger.info(f"Loading model '{MODEL_NAME}'...")
        global classifier
        classifier = pipeline("sentiment-analysis", model=MODEL_NAME)
        app.state.model_ready = True
        logger.info(f"Model ready.")

    except Exception as e:
        logger.error(f"Error: {e}")

# Load web application:
app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

# Load model:
load_model()

@app.get("/ready")
async def health_check():
    """
    Readiness endpoint: until the model is not ready, the workload cannot process any requests.
    """
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail="Model loading...")
    return {"status": "ready"}

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    """
    Inferences endpoint.
    """
    if classifier is None:
        logger.warning(f"Model not ready on predict request.")
        raise HTTPException(status_code=503, detail="Model not ready")

    # Inference:
    prediction = classifier(request.text)[0]
    label = prediction['label']
    score = prediction['score']
    
    return {"text": request.text, "sentiment": label, "confidence": score}

if __name__ == "__main__":
    uvicorn.run("main:app", host=NETWORK_INTERFACE, port=PORT, reload=True)