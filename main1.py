from fastapi import FastAPI, UploadFile, File
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import nltk
from docx import Document
import io

app = FastAPI()

w2v = KeyedVectors.load_word2vec_format(r"D:\gitab\vi_txt\vi.vec", binary=False)
vocab = w2v.key_to_index

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@app.post("/summarize_docx")
async def summarize_docx(file: UploadFile = File(...), n_clusters: int = 5):
    contents = await file.read()
    
    # Đọc nội dung file docx
    doc = Document(io.BytesIO(contents))
    text = '\n'.join([para.text for para in doc.paragraphs])

    # Xử lý văn bản
    contents_parsed = text.lower().replace('\n', '. ').strip()
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

    if len(X) < n_clusters:
        return {"error": "Số cụm lớn hơn số câu, vui lòng giảm n_clusters."}

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    avg = [np.mean(np.where(kmeans.labels_ == j)[0]) for j in range(n_clusters)]
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])

    return {"summary": summary}
