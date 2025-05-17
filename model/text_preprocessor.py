import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\b\w+\b')
        self.stop_words = set(stopwords.words('portuguese'))

    def preprocess(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        return ' '.join([
            token for token in tokens
            if token.isalpha() and token not in self.stop_words
        ])

    def preprocess_list(self, texts):
        return [self.preprocess(text) for text in texts]
