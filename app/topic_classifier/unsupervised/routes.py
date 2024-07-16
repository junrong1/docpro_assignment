import logging
from fastapi import APIRouter, HTTPException
from app.topic_classifier.unsupervised.classifier import UnsupervisedClassifier
from app.topic_classifier.unsupervised.models import *

classifier = UnsupervisedClassifier()

router = APIRouter()


@router.post("/gen_unsup_topics",
             tags=["Topic Classifier"],
             summary="Generate topics for target content by unsupervised learning")
def get_topics(
        request: ClassifierRequest,
) -> ClassifierResponse:
    try:
        return classifier.gen_response(request)
    except Exception as e:
        logging.error(f"Error with the classifier {e}")
        raise HTTPException(status_code=500, detail="Classifier went wrong")
