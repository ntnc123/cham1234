from typing import Union
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {item_id: item_id, "q":q}


from pydantic import BaseModel
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import nltk

w2v = KeyedVectors.load_word2vec_format(r"D:\gitab\vi_txt\vi.vec", binary=False)
vocab = w2v.key_to_index

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextRequest(BaseModel): 
    content: str
    n_clusters: int = 5

@app.post("/summarize")
async def summarize_text(request: TextRequest):
    contents_parsed = request.content.lower().replace('\n', '. ').strip()
    sentences = nltk.sent_tokenize(contents_parsed)

    X = []
    for sentence in sentences:
        sentence_tokenized = ViTokenizer.tokenize(sentence)
        words = sentence_tokenized.split(" ")
        sentence_vec = np.zeros((100))
        for word in words:
            if word in vocab:
                sentence_vec += w2v[word]
        X.append(sentence_vec)

    if len(X) < request.n_clusters:
        return {"error": "Số cụm lớn hơn số câu, vui lòng giảm n_clusters."}

    kmeans = KMeans(n_clusters=request.n_clusters, random_state=0).fit(X)
    avg = [np.mean(np.where(kmeans.labels_ == j)[0]) for j in range(request.n_clusters)]
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(request.n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])

    return {"summary": summary}
