#Bước 1: Gán văn bản cần tóm tắt vào biến content
content = '''
Tối nào cũng vậy, cứ đến lúc con bé lớn ông Hai thu que đóm cháy lập lòe trong chiếc nón rách tất tả đi từ nhà bếp lên, và bà Hai ngồi ngây thuỗn cái mặt trước đĩa đèn dầu lạc, lầm bầm tính toán những tiền cua, tiền bún, tiền chuối, tiền kẹo… thì ông Hai vùng dậy, sang bên bác Thứ nói chuyện. Không hiểu sao cứ đến lúc ấy ông Hai lại thấy buồn. Nằm nghe tiếng súng dội trong đêm tối và nhất là cái tiếng rì rầm tính toán tiền nong của mụ vợ, tự nhiên ông sinh ra nghĩ ngợi vẩn vơ, nó bực dọc làm sao ấy. Mà ông, thì không thích nghĩ ngợi như thế một tí nào. ông vốn là người hay làm, ở quê ông làm suốt ngày, không mấy lúc chịu ngơi chân ngơi tay. Không đi cày đi cuốc, không gánh phân tát nước thì ông cũng phải bày vẽ ra công việc gì để làm: đan rổ, đan rá hay chữa cái chuồng gà, cạp lại tấm liếp. Từ ngày tản cư lên đây, suốt ngày mấy bố con nhong nhóng ngồi ăn, tối đến lại nghe những tiếng rì rầm tính toán ấy, ruột gan ông cứ nóng lên như lửa đốt. Ông phải đi chơi cho khuây khỏa. Lần nào cũng như lần nào, cứ vừa nhô đầu qua cái mái lá bên gian bác Thứ là ông lão hỏi ngay: “Thế nào, hôm nay có gì không bác?”...
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