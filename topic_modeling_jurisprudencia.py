import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Downloads iniciais
nltk.download('punkt')
nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\b\w+\b')

# Carrega o dataset do Hugging Face
dataset = load_dataset("joelniklaus/brazilian_court_decisions", split="train")
df = pd.DataFrame(dataset)

# Usa a coluna correta: 'ementa_text'
df = df.dropna(subset=['ementa_text'])
texts = df['ementa_text'].astype(str).tolist()

# Pré-processamento básico
stop_words = set(stopwords.words('portuguese'))
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    return ' '.join([w for w in tokens if w.isalpha() and w not in stop_words])

texts_clean = [preprocess(t) for t in texts]

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
X = vectorizer.fit_transform(texts_clean)

# Modelos de Tópicos
n_topics = 5

# NMF
nmf = NMF(n_components=n_topics, random_state=42)
nmf_topics = nmf.fit_transform(X)

# LDA
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_topics = lda.fit_transform(X)

# Mostrar tópicos
def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTópico {topic_idx+1}:")
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

print("\n--- Tópicos via NMF ---")
display_topics(nmf, vectorizer.get_feature_names_out())

print("\n--- Tópicos via LDA ---")
display_topics(lda, vectorizer.get_feature_names_out())
