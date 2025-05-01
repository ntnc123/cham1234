#Thư viện
#!pip install numpy==1.26.4
# pip install nltk
# pip install gensim
# pip install scikit-learn
import numpy as np
# print(np.__version__)

import nltk
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Bước 1: Tiền xử lý văn bản và tách câu
def preprocess_text(text):
    text = text.lower().strip()
    # Tách câu dựa trên dấu chấm, dấu hỏi, dấu cảm
    sentences = [sent.strip() for sent in nltk.regexp_tokenize(text, pattern=r'[^.!?]+')]
    return sentences

# Bước 2: Vector hóa từ và câu
def train_word2vec(sentences):
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    model = Word2Vec(sentences=tokenized_sentences, vector_size=128, window=5, min_count=1)
    return model, tokenized_sentences

def sentence_vector(model, tokenized_sentence):
    vectors = [model.wv[word] for word in tokenized_sentence if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Bước 3: Phân cụm
def cluster_sentences(sent_vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(sent_vectors)
    return kmeans.labels_, kmeans.cluster_centers_

# Bước 4: Tìm câu đại diện mỗi cụm
def find_representative_sentences(sent_vectors, labels, centers):
    representatives = []
    for i in range(len(centers)):
        indices = [j for j, label in enumerate(labels) if label == i]
        if indices:
            sim = cosine_similarity([centers[i]], [sent_vectors[j] for j in indices])
            best_index = indices[np.argmax(sim)]
            representatives.append(best_index)
    return sorted(representatives)

# Bước 5: Sắp xếp và sinh đoạn tóm tắt
def generate_summary(sentences, rep_indices):
    summary = [sentences[i] for i in rep_indices]
    return ' '.join(summary)

text = """
Văn bản là một loại hình phương tiện để ghi nhận, lưu giữ và truyền đạt các thông tin từ chủ thể này sang chủ thể khác bằng ký hiệu gọi là chữ viết. Nó gồm tập hợp các câu có tính trọn vẹn về nội dung, hoàn chỉnh về hình thức, có tính liên kết chặt chẽ và hướng tới một mục tiêu giao tiếp nhất định. Hay nói khác đi, văn bản là một dạng sản phẩm của hoạt động giao tiếp bằng ngôn ngữ được thể hiện ở dạng viết trên một chất liệu nào đó (giấy, bia đá,...)
"""

# Chạy từng bước
sentences = preprocess_text(text)
model, tokenized_sentences = train_word2vec(sentences)
sent_vectors = np.array([sentence_vector(model, sent) for sent in tokenized_sentences])
n_clusters = min(3, len(sentences))  # không phân cụm nhiều hơn số câu
labels, centers = cluster_sentences(sent_vectors, n_clusters)
rep_indices = find_representative_sentences(sent_vectors, labels, centers)
summary = generate_summary(sentences, rep_indices)

print("=== TÓM TẮT ===")
print(summary)