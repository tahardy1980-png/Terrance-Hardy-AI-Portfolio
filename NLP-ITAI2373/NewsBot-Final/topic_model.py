import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords')

# Sample dataset (we’ll replace this later with real news data)
documents = [
    "AI is transforming technology and machine learning models",
    "The government passed a new election reform bill",
    "Sports teams are preparing for the championship game",
    "Machine learning and AI are growing rapidly in tech",
    "Political debates are heating up during election season"
]

def run_topic_modeling(docs, n_topics=2):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()

    topics = []

    for i, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-5:]]
        topics.append((f"Topic {i+1}", top_words))

    return topics


if __name__ == "__main__":
    results = run_topic_modeling(documents)

    for topic, words in results:
        print(topic, ":", words)
