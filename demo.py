#Thư viện
#pip install python-docx
import os
import nltk
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

nltk.download("punkt")

# ===== Bước 1: Tách câu và token hóa tiếng Việt =====
def preprocess_text(text):
    text = text.strip()
    # Tách câu: dùng NLTK với hỗ trợ tiếng Việt (đơn giản)
    sentences = nltk.sent_tokenize(text, language='english')  # vẫn hiệu quả với tiếng Việt đơn giản
    return sentences

def tokenize_sentences(sentences):
    return [nltk.word_tokenize(sent.lower()) for sent in sentences]

# ===== Bước 2: Train hoặc load mô hình Word2Vec =====
def train_or_load_word2vec(all_tokenized_sentences, model_path="word2vec_vi.model"):
    if os.path.exists(model_path):
        print("🔁 Loading Word2Vec model từ file...")
        model = Word2Vec.load(model_path)
    else:
        print("🧠 Training Word2Vec model...")
        model = Word2Vec(sentences=all_tokenized_sentences, vector_size=128, window=5, min_count=1)
        model.save(model_path)
    return model

def sentence_vector(model, tokenized_sentence):
    vectors = [model.wv[word] for word in tokenized_sentence if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# ===== Bước 3: Phân cụm và chọn câu đại diện =====
def cluster_sentences(sent_vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(sent_vectors)
    return kmeans.labels_, kmeans.cluster_centers_

def find_representative_sentences(sent_vectors, labels, centers):
    representatives = []
    for i in range(len(centers)):
        indices = [j for j, label in enumerate(labels) if label == i]
        if indices:
            sims = cosine_similarity([centers[i]], [sent_vectors[j] for j in indices])
            best_index = indices[np.argmax(sims)]
            representatives.append(best_index)
    return sorted(representatives)

def generate_summary(sentences, rep_indices):
    return ' '.join([sentences[i] for i in rep_indices])

# ===== Bước 4: Tóm tắt một văn bản =====
def summarize_text(text, model):
    sentences = preprocess_text(text)
    tokenized = tokenize_sentences(sentences)
    sent_vectors = np.array([sentence_vector(model, sent) for sent in tokenized])
    n_clusters = min(3, len(sentences))  # số cụm tối đa là số câu
    labels, centers = cluster_sentences(sent_vectors, n_clusters)
    rep_indices = find_representative_sentences(sent_vectors, labels, centers)
    return generate_summary(sentences, rep_indices)

#documents = [
 #   """Ôtô tải bị cấm vào 5 tuyến đường ở huyện Bình Chánh trong 6 ngày để đảm an toàn khi tổ chức đại lễ phật đản Vesak...""",
#    """Theo Sở Xây dựng TP HCM, thời gian cấm xe tải áp dụng từ ngày 3 đến 8/5...""",
#]
# Tạo danh sách văn bản để tóm tắt
from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    full_text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
    return full_text

file_path = "taytien.docx"
text = read_docx(file_path)  # Đọc toàn bộ file Word
sentences = preprocess_text(text)
tokenized = tokenize_sentences(sentences)


# Train hoặc load Word2Vec
model = train_or_load_word2vec(tokenized)


# Tóm tắt
summary = summarize_text(text, model)

print("\n=== TÓM TẮT ===")
print(summary)
