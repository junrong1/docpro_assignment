from fastapi import FastAPI
from app.topic_classifier.unsupervised.routes import router as unsupervised_classifier
from app.topic_classifier.supervised.routes import router as supervised_classifier
from app.health import router as health_router

app = FastAPI()
app.include_router(health_router)
app.include_router(unsupervised_classifier)
app.include_router(supervised_classifier)
