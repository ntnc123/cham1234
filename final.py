#THƯ VIỆN
# pip install nltk
# pip install gensim 
# pip install scikit-learn 
# pip install pyvi 
# pip install numpy==1.26.4 

content = '''
Có được tờ phụ san tái hiện những khoảnh khắc vàng của lịch sử trong không khí cả nước chào mừng 50 năm thống nhất đất nước, với nhiều độc giả Bạc Liêu, đó là hạnh phúc. Hạnh phúc của những công dân yêu nước, biết trân trọng những giá trị lịch sử của người đi trước!
Và đó cũng là mục tiêu của Báo Nhân Dân - tờ báo được xem là anh cả của nền báo chí cách mạng Việt Nam - khi mở đợt thông tin đặc biệt nhân kỷ niệm 50 năm Ngày Giải phóng miền Nam, thống nhất đất nước vừa qua. Những phần việc thiết thực đã “phát huy vai trò của báo chí trong nhiệm vụ góp phần tuyên truyền, giáo dục truyền thống yêu nước, tinh thần đại đoàn kết dân tộc của mọi thế hệ người Việt Nam, góp phần tạo động lực mạnh mẽ để đất nước ta bước vào kỷ nguyên mới - kỷ nguyên phát triển phồn vinh, thịnh vượng” là ghi nhận của Phó Chủ tịch nước - Võ Thị Ánh Xuân.
Một trong những phần việc ấy là Báo Nhân Dân đã in và phát hành miễn phí hàng trăm ngàn bản phụ san đặc biệt này đến độc giả cả nước. Đó là những tư liệu đặc biệt về Chiến dịch Hồ Chí Minh được thiết kế và in ấn trên khổ giấy Báo Nhân Dân, gồm 8 trang A3. “Cuốn” độc giả ở chỗ, phụ san tích hợp mã QR để người xem dễ dàng truy cập và theo dõi nhiều video tư liệu trên các nền tảng mạng xã hội.
Đổi mới cách thức tuyên truyền để hướng đến những đối tượng độc giả thời công nghệ số là nhiệm vụ mới và thường trực mà báo chí hiện đại phải nghĩ đến. Những câu chuyện lịch sử được học từ ghế nhà trường đã quá quen thuộc, nên rất cần sự trải nghiệm mới mẻ hơn từ báo chí. Phải nhìn nhận rằng, báo chí truyền thống đầy ắp những tuyến bài đặc sắc, nhưng đôi khi sẽ khó hòa nhập với thị hiếu giới trẻ thời công nghệ số. Vì vậy, rất cần có những hình thức phù hợp để báo chí xây dựng hệ giá trị về lòng yêu nước, tự hào dân tộc - một nhiệm vụ quan trọng để báo chí góp phần “xây dựng, phát triển văn hóa, con người đáp ứng yêu cầu phát triển bền vững của đất nước” trong bối cảnh mới.
'''

# Đọc nội dung từ file
from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)
    

#Bước 1: Tiền xử lý văn bản
contents_parsed = content.lower() 
contents_parsed = contents_parsed.replace('\n', '. ') 
contents_parsed = contents_parsed.strip() 
#In ra nội dung sau khi tiền xử lý
# print("Nội dung sau khi tiền xử lý:\n",contents_parsed) 


#Bước 2: Tách câu trong văn bản
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

sentences = nltk.sent_tokenize(contents_parsed)
#In ra danh sách các câu đã tách
# print("Danh sách câu:\n",sentences) 


#Bước 3: Tách từ trong câu và tính toán vector cho từng câu
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format(r"D:\gitab\vi_txt\vi.vec", binary=False)
vocab = w2v.key_to_index

from pyvi import ViTokenizer

import numpy as np
X = []
for sentence in sentences:
    sentence_tokenized = ViTokenizer.tokenize(sentence)
    words = sentence_tokenized.split(" ")
    sentence_vec = np.zeros((100))
    for word in words:
        if word in vocab:
            sentence_vec += w2v[word]
    X.append(sentence_vec)


#Bước 4: Phân cụm câu bằng KMeans
from sklearn.cluster import KMeans
n_clusters = 3
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