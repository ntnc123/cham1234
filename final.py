content = '''
Quang Dũng là một người nghệ sĩ đa tài, ông là một nhạc sĩ, một họa sĩ, đồng thời cũng là một thi sĩ. Chính vì vậy mà trong những vần thơ của Quang Dũng, ta thấy có cả âm hưởng của nhạc và họa. Ông được mệnh danh là “bóng mây qua đỉnh Việt”, với hồn thơ bay bổng, phóng khoáng, hào hoa đầy lãng mạn và tha thiết tình cảm dành cho bạn bè, quê hương xứ sở. Dù viết ít, viết theo cảm hứng nhưng ông đã thực sự thành công khi viết về người lính, và “Tây Tiến” là bài thơ tiên phong cho phong cách thơ Quang Dũng, đồng thời cũng là đứa con tinh thần tráng kiện và hào hoa, là đỉnh cao trong sự nghiệp sáng tác của ông. Với hai đặc điểm nổi bật là cảm hứng lãng mạn và tinh thần bi tráng trong bài thơ, Quang Dũng đã khắc họa thành công hình tượng người lính Tây Tiến trên cái nền cảnh thiên nhiên núi rừng miền Tây hùng vĩ, tráng lệ.
Tây Tiến là tên của một đơn vị quân đội thành lập vào năm 1947, có nhiệm vụ phối hợp với bộ đội Lào, bảo vệ biên giới Việt – Lào và đánh tiêu hao lực lượng quân đội Pháp ở Thượng Lào cũng như miền Tây Bắc Bộ Việt Nam. Đoàn quân Tây Tiến sau một thời gian hoạt động ở Lào đã trở về Hòa Bình thành lập trung đoàn 52. Cuối năm 1948, Quang Dũng chuyển sang đơn vị khác, và cũng chính thời gian này, bài thơ “Tây Tiến” ra đời. “Tây Tiến” được viết trong niềm thương nỗi nhớ đơn vị cũ, chiến trường xưa. Ban đầu tác phẩm có tên là “Nhớ Tây Tiến”, sau này khi in trong tập “Mây đầu ô”, tác giả đổi lại thành “Tây Tiến”. Toàn bài thơ bao trùm bởi những nỗi nhớ: đó là nỗi nhớ về những chặng đường hành quân, nỗi nhớ về những kỉ niệm nơi miền Tây, nỗi nhớ về binh đoàn Tây Tiến anh hùng và nỗi nhớ về lời thề của binh đoàn Tây Tiến.
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
w2v = KeyedVectors.load_word2vec_format(r"D:\gitab\vi_txt\vi.vec")
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

#Bước 4: Phân cụm câu băng KMeans
from sklearn.cluster import KMeans
n_clusters = 5 #Sô lượng cụm (có thể thay đổi)
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)

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