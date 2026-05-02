import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Step 1: lowercase
    text = text.lower()

    # Step 2: remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 3: tokenize
    tokens = word_tokenize(text)

    # Step 4: remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]

    return filtered


if __name__ == "__main__":
    sample = "AI is changing how news is analyzed and understood."
    print(preprocess_text(sample))
