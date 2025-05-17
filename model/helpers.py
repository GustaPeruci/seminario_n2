import matplotlib.pyplot as plt
import numpy as np

def extract_topics(model, feature_names, top_n=10):
    topics = {}
    for idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-top_n - 1:-1]]
        topics[idx] = top_words
    return topics

def plot_topic_weights(W):
    topic_strength = np.mean(W, axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(topic_strength)), topic_strength)
    ax.set_title("Importância Média de Cada Tópico")
    ax.set_xlabel("Tópico")
    ax.set_ylabel("Peso Médio")
    return fig
