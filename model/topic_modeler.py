import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import matplotlib.pyplot as plt

from model.text_preprocessor import TextPreprocessor
from model.helpers import extract_topics, plot_topic_weights

class TopicModeler:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        self.texts_clean, self.feature_names = self._prepare_dataset()

    def _prepare_dataset(self):
        datasets = [
            load_dataset("joelniklaus/brazilian_court_decisions", split=split)
            for split in ["train", "test", "validation"]
        ]
        df = pd.concat([pd.DataFrame(ds) for ds in datasets], ignore_index=True)
        df = df.dropna(subset=["ementa_text"])
        texts = df["ementa_text"].astype(str).tolist()
        texts_clean = self.preprocessor.preprocess_list(texts)
        X = self.vectorizer.fit_transform(texts_clean)
        return X, self.vectorizer.get_feature_names_out()

    def run(self, model_type: str, n_topics: int):
        if model_type == "nmf":
            model = NMF(n_components=n_topics, random_state=42)
        elif model_type == "lda":
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        else:
            raise ValueError("Modelo inválido: escolha 'nmf' ou 'lda'.")

        W = model.fit_transform(self.texts_clean)
        topics = extract_topics(model, self.feature_names, top_n=10)
        fig = plot_topic_weights(W)
        return topics, fig
