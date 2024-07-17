from app.topic_classifier.unsupervised.models import *
from typing import List
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
import logging
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
import openai
import os


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


class UnsupervisedClassifierV2:

    def __init__(self):
        pass

    def _label_classifier(self, content: List[str]) -> ClassifierResponse:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(content, show_progress_bar=True, batch_size=32)

        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=20, ngram_range=(1, 2))
        ctfidf_model = ClassTfidfTransformer()

        # KeyBERT
        keybert_model = KeyBERTInspired()

        # Part-of-Speech
        pos_model = PartOfSpeech("en_core_web_sm")

        # MMR
        mmr_model = MaximalMarginalRelevance(diversity=0.3)

        # GPT-3.5
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = """
        I have a topic that contains the following documents: 
        [DOCUMENTS]
        The topic is described by the following keywords: [KEYWORDS]

        Based on the information above, extract a short but highly descriptive topic label of at most 3 words.
        Please do not generate the label too specific, these are some example labels, for you to understand the principle
        Example labels: Clothing and accessories, Household goods, Devices and electronics, Adult products 
        Make sure it is in the following format:
        topic: <topic label>
        """
        openai_model = OpenAI(client, model="gpt-4o", exponential_backoff=True, chat=True, prompt=prompt)

        # All representation models
        representation_model = {
            "KeyBERT": keybert_model,
            "OpenAI": openai_model,  # Uncomment if you will use OpenAI
            "MMR": mmr_model,
            "POS": pos_model
        }

        topic_model = BERTopic(

            # Pipeline models
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,

            # Hyperparameters
            top_n_words=10,
            verbose=True
        )

        # Train model
        topics, probs = topic_model.fit_transform(list(map(str, content)), embeddings)

        # Filter outlier
        new_topics = topic_model.reduce_outliers(content, topics, strategy="embeddings", embeddings=embeddings)
        topic_labels = [topic_model.get_topic(x, full=True)["OpenAI"][0][0] for x in new_topics]
        return ClassifierResponse(org_content=content, topics=topic_labels)

    def gen_response(self, request: ClassifierRequest) -> ClassifierResponse:
        return self._label_classifier(
            request.query
        )
