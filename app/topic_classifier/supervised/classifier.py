from openai import OpenAI
from app.topic_classifier.supervised.models import *
from app.topic_classifier.supervised.prompt import SYSTEM_PROMPT
import os
import math


class SupervisedClassifier:

    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', None))
        self.label_dict = {
            "sports": 0,
            "technology": 1,
            "politics": 2,
            "entertainment": 3
        }

    def _label_classifier(self, content: List[str]) -> ClassifierResponse:
        gpt_label = []
        perplexity = []
        for c in content:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                seed=42,
                logprobs=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"content: {c}\nlabel:"},
                ]
            )
            # Use perplexity to control the outlier label
            logprobs = [token.logprob for token in response.choices[0].logprobs.content]
            perplexity.append(math.exp(-sum(logprobs)/len(logprobs)))
            gpt_label.append(response.choices[0].message.content)

        topic_labels = self._error_handler(gpt_label)
        return ClassifierResponse(org_content=content, topics=topic_labels)

    def _error_handler(self, gpt_label: List[str], perplexity: List[float], threshold=0.1) -> List[str]:
        final_res = []
        for label, score in zip(gpt_label, perplexity):
            if label not in self.label_dict:
                final_res.append("other")
            else:
                if 1 - score > threshold:
                    final_res.append("other")
                else:
                    final_res.append(label)
        return final_res

    def gen_response(self, request: ClassifierRequest) -> ClassifierResponse:
        return self._label_classifier(
            request.query
        )

