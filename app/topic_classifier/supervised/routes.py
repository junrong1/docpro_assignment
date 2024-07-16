import logging
from fastapi import APIRouter, HTTPException
from app.topic_classifier.supervised.classifier import SupervisedClassifier
from app.topic_classifier.supervised.models import *

classifier = SupervisedClassifier()

router = APIRouter()


@router.post("/gen_sup_topics",
             tags=["Topic Classifier"],
             summary="Generate topics for target content by supervised learning GPT")
def get_topics(
        request: ClassifierRequest,
) -> ClassifierResponse:
    try:
        return classifier.gen_response(request)
    except Exception as e:
        logging.error(f"Error with the classifier {e}")
        raise HTTPException(status_code=500, detail="Classifier went wrong")
