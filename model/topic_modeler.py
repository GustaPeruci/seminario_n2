import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from model.text_preprocessor import TextPreprocessor
from model.helpers import extract_topics, plot_topic_weights

class TopicModeler:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
        self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.texts_clean, self.feature_names, self.raw_texts = self._prepare_dataset()

    def _prepare_dataset(self):
        datasets = [
            load_dataset("joelniklaus/brazilian_court_decisions", split=split)
            for split in ["train", "test", "validation"]
        ]
        df = pd.concat([pd.DataFrame(ds) for ds in datasets], ignore_index=True)
        df = df.dropna(subset=["ementa_text"])
        raw_texts = df["ementa_text"].astype(str).tolist()
        texts_clean = self.preprocessor.preprocess_list(raw_texts)
        X = self.vectorizer.fit_transform(texts_clean)
        return X, self.vectorizer.get_feature_names_out(), raw_texts

    def generate_topic_title(self, topic_words, topic_docs):
        # Get embeddings for topic words and documents
        word_embeddings = self.sentence_model.encode([' '.join(topic_words)])
        doc_embeddings = self.sentence_model.encode(topic_docs[:5])
        
        # Find most representative document
        similarities = cosine_similarity(word_embeddings, doc_embeddings)
        most_rep_idx = similarities.argmax()
        
        # Extract key phrase from the document
        sentences = topic_docs[most_rep_idx].split('.')
        if sentences:
            return sentences[0][:50] + '...' if len(sentences[0]) > 50 else sentences[0]
        return ' '.join(topic_words[:3])

    def run(self, model_type: str, n_topics: int):
        if model_type == "nmf":
            model = NMF(n_components=n_topics, random_state=42)
        elif model_type == "lda":
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        else:
            raise ValueError("Modelo inválido: escolha 'nmf' ou 'lda'.")

        W = model.fit_transform(self.texts_clean)
        topic_dict = extract_topics(model, self.feature_names, top_n=10)
        
        # Generate titles using document context
        for idx, words in topic_dict.items():
            topic_docs = [doc for doc, weights in zip(self.raw_texts, W) 
                         if weights[idx] == max(weights)]
            topic_dict[idx] = {
                'words': words,
                'title': self.generate_topic_title(words, topic_docs),
                'weight': float(np.mean(W[:, idx]))
            }
            
        return topic_dict, plot_topic_weights(W)
