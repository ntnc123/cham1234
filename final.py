#Bước 1: Gán văn bản cần tóm tắt vào biến content
content = '''
Overfitting là hiện tượng mô hình tìm được quá khớp
với dữ liệu training. Việc quá khớp này có thể dẫn đến
việc dự đoán nhầm nhiễu, và chất lượng mô hình không
còn tốt trên dữ liệu test nữa. Dữ liệu test được giả sử là
không được biết trước, và không được sử dụng để xây
dựng các mô hình Machine Learning. Vấn đề quá vừa
dữ liệu trong học máy ảnh hưởng đến độ chính xác của
các kỹ thuật học máy.
'''

#Bước 2: Tiền xử lý văn bản
contents_parsed = content.lower() #Biến đổi hết thành chữ thường
contents_parsed = contents_parsed.replace('\n', '. ') #Đổi các ký tự xuống dòng thành chấm câu
contents_parsed = contents_parsed.strip() #Loại bỏ đi các khoảng trắng thừa

#Bước 3: Tách câu trong văn bản
import nltk
nltk.download('punkt_tab')
sentences = nltk.sent_tokenize(contents_parsed)

#Bước 4: Tách từ trong câu
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format(r"D:\gitab\vi_txt\vi.vec")
vocab = w2v.key_to_index  # Danh sách các từ trong từ điển
from pyvi import ViTokenizer

#Bước 5: Tính toán vector cho từng câu
import numpy as np
X = []
for sentence in sentences:
    sentence_tokenized = ViTokenizer.tokenize(sentence)
    words = sentence_tokenized.split(" ")
    sentence_vec = np.zeros((100))  # 100 là kích thước vector, thay bằng kích thước thực tế của bạn
    for word in words:
        if word in vocab:
            sentence_vec += w2v[word]  # Truy cập vector từ trực tiếp bằng w2v[word]
    X.append(sentence_vec)

#Bước 6: Phân cụm câu băng KMeans
from sklearn.cluster import KMeans
n_clusters = 5 #Sô lượng cụm (có thể thay đổi)
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)

#Bước 7: Tìm câu đại diện cho từng cụm
from sklearn.metrics import pairwise_distances_argmin_min

avg = []
for j in range(n_clusters):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append(np.mean(idx))
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
ordering = sorted(range(n_clusters), key=lambda k: avg[k])
summary = ' '.join([sentences[closest[idx]] for idx in ordering])

#Bước 8: In ra kết quả
print(summary)