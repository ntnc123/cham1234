content = '''Nội dung cốt lõi Tư tưởng Hồ Chí Minh
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
'''


#Bước 1: Tiền xử lý văn bản
contents_parsed = content.lower() #Biến đổi hết thành chữ thường
contents_parsed = contents_parsed.replace('\n', '. ') #Đổi các ký tự xuống dòng thành chấm câu
contents_parsed = contents_parsed.strip() #Loại bỏ đi các khoảng trắng thừa
#In ra nội dung sau khi tiền xử lý
# print("Nội dung sau khi tiền xử lý:\n",contents_parsed) 


#Bước 2: Tách câu trong văn bản
import nltk
nltk.download('punkt_tab')
sentences = nltk.sent_tokenize(contents_parsed)
#In ra danh sách các câu đã tách
# print("Danh sách câu:\n",sentences) 


#Bước 3: Tách từ trong câu và tính toán vector cho từng câu
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format(r"D:\gitab\vi_txt\vi.vec", binary=False)
vocab = w2v.key_to_index  # Danh sách các từ trong từ điển
from pyvi import ViTokenizer

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


#Bước 4: Phân cụm câu bằng KMeans
from sklearn.cluster import KMeans
n_clusters = 3 #Sô lượng cụm (có thể thay đổi)
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)

#In ra nhãn của từng câu
print("Label vector:", kmeans.labels_)


#Bước 5: Tìm câu đại diện cho từng cụm
from sklearn.metrics import pairwise_distances_argmin_min

avg = []
for j in range(n_clusters):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append(np.mean(idx))
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
ordering = sorted(range(n_clusters), key=lambda k: avg[k])
summary = ' '.join([sentences[closest[idx]] for idx in ordering])


#Bước 6: In ra kết quả
print("\nTóm Tắt:\n",summary)