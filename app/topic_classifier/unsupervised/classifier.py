from app.topic_classifier.unsupervised.models import *
from typing import List
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import logging


class UnsupervisedClassifier:

    def __init__(self):
        self.label_dict = {
            "sports": 0,
            "technology": 1,
            "politics": 2,
            "entertainment": 3
        }

    def _label_classifier(self, content: List[str]) -> ClassifierResponse:
        # We might do not have enough data
        # First step is to use clustering model to do zero-shot classification
        # After getting more and more data, we can change to supervised

        # Init an embedding model
        logging.info(f"Start label classifier with {len(content)}")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("Loading embedding model successfully")

        # Init a dimension reduction model
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        logging.info("Loading dimension reduction model successfully")

        zeroshot_topic_list = list(self.label_dict.keys())

        # All the hyper parameter might need tune for specific cases
        logging.info("Topic model processing....")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            min_topic_size=10,
            zeroshot_topic_list=zeroshot_topic_list,
            zeroshot_min_similarity=.6,
            representation_model=KeyBERTInspired()
        )

        logging.info("Finished the classification")
        topics, _ = topic_model.fit_transform(content)
        # TODO Reduce outlier
        topic_labels = self._parse_topic(topics, topic_model)
        return ClassifierResponse(org_content=content, topics=topic_labels)

    def _parse_topic(self, topics: List[int], topic_model: BERTopic) -> List[str]:
        topic_label = []
        topic_info_df = topic_model.get_topic_info()
        for topic_index in topics:
            # -1 is the outlier
            if topic_index == -1:
                topic_label.append('other')
            else:
                topic_info = topic_info_df.loc[topic_index + 1, :]
                name = topic_info["Name"]
                if name in list(self.label_dict.keys()):
                    topic_label.append(name)
                else:
                    # process the other topic
                    for target in list(self.label_dict.keys()):
                        if name.lower().__contains__(target.lower()):
                            topic_label.append(name)
                            break
                        else:
                            topic_label.append("other")
        return topic_label

    def gen_response(self, request: ClassifierRequest) -> ClassifierResponse:
        return self._label_classifier(
            request.query
        )
