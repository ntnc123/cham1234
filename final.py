#THƯ VIỆN
# pip install nltk #Sử dụng để tách câu
# pip install gensim #Sử dụng để tách từ tiếng Việt
# pip install sklearn #Phân cụm câu và chọn câu đại diện
# pip install pyvi #Tải mô hình Word2Vec và truy xuất vector từ
# pip install numpy==1.26.4 #Tính toán trung bình vector câu

content = '''
Tối nào cũng vậy, cứ đến lúc con bé lớn ông Hai thu que đóm cháy lập lòe trong chiếc nón rách tất tả đi từ nhà bếp lên, và bà Hai ngồi ngây thuỗn cái mặt trước đĩa đèn dầu lạc, lầm bầm tính toán những tiền cua, tiền bún, tiền chuối, tiền kẹo… thì ông Hai vùng dậy, sang bên bác Thứ nói chuyện. Không hiểu sao cứ đến lúc ấy ông Hai lại thấy buồn. Nằm nghe tiếng súng dội trong đêm tối và nhất là cái tiếng rì rầm tính toán tiền nong của mụ vợ, tự nhiên ông sinh ra nghĩ ngợi vẩn vơ, nó bực dọc làm sao ấy. Mà ông, thì không thích nghĩ ngợi như thế một tí nào. Ông vốn là người hay làm, ở quê ông làm suốt ngày, không mấy lúc chịu ngơi chân ngơi tay. Không đi cày đi cuốc, không gánh phân tát nước thì ông cũng phải bày vẽ ra công việc gì để làm: đan rổ, đan rá hay chữa cái chuồng gà, cạp lại tấm liếp. Từ ngày tản cư lên đây, suốt ngày mấy bố con nhong nhóng ngồi ăn, tối đến lại nghe những tiếng rì rầm tính toán ấy, ruột gan ông cứ nóng lên như lửa đốt. Ông phải đi chơi cho khuây khỏa. Lần nào cũng như lần nào, cứ vừa nhô đầu qua cái mái lá bên gian bác Thứ là ông lão hỏi ngay: “Thế nào, hôm nay có gì không bác?”...
'''

# Đọc nội dung từ file
# from docx import Document

# def read_docx(file_path):
#     doc = Document(file_path)
#     full_text = []
#     for paragraph in doc.paragraphs:
#         full_text.append(paragraph.text)
#     return '\n'.join(full_text)\
    
# content = read_docx(r"D:\gitab\TT3601_NguyenThiNgocCham_BaiChinhLuan.docx")

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
n_clusters = 5 #Sô lượng cụm (có thể thay đổi)
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

