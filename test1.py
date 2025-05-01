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
Nội dung cốt lõi Tư tưởng Hồ Chí Minh
Tư tưởng về giải phóng dân tộc, giải phóng giai cấp, giải phóng con người;
Độc lập dân tộc gắn liền với chủ nghĩa xã hội; kết hợp sức mạnh dân tộc với sức mạnh thời đại;
Sức mạnh của nhân dân, của khối đại đoàn kết dân tộc;
Quyền làm chủ của nhân dân, xây dựng Nhà nước thật sự của dân, do dân và vì dân
Quốc phòng toàn dân, xây dựng lực lượng vũ trang nhân dân;
Về phát triển kinh tế, văn hóa, không ngừng nâng cao đời sống vật chất và tinh thần của nhân dân;
Về phát triển kinh tế và văn hóa, không ngừng nâng cao đời sống vật chất và tinh thần của nhân dân;
Đạo đức cách mạng, cần, kiệm, liêm, chính, chí công vô tư;
Chăm lo bồi dưỡng thế hệ cách mạng cho đời sau;
Xây dựng Đảng trong sạch, vững mạnh, cán bộ, Đảng viên vừa là lãnh đạo, vừa là người đầy tớ thật trung thành của nhân dân...
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