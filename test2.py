import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from gensim.models import KeyedVectors

# ==== NẠP MÔ HÌNH WORD2VEC ====
w2v_model = KeyedVectors.load_word2vec_format("vi_txt/vi.vec") # Đổi thành đường dẫn mô hình của bạn

# ==== DỮ LIỆU ĐẦU VÀO ====
original_text = '''
Tràng là một người dân sống ngụ cư sống cùng với mẹ già. Anh làm nghề kéo xe bò thuê. Một lần trên đường kéo cái xe bò thóc trên tỉnh anh quen được Thị, Chỉ với bốn bát bánh đúc thị đã đồng ý làm vợ Tràng. Về đến nhà Tràng phấp phỏng chờ mẹ về để thưa chuyện. Đến khi bà cụ Tứ trở về vô cùng ngạc nhiên khi thấy có người phụ nữ lạ trong nhà, nghe con kể rõ sự tình người mẹ nghèo khổ ấy đã hiểu ra và  chấp nhận nàng dâu mới động viên các con cố gắng làm ăn. Sáng hôm sau Tràng thức dậy thấy mọi thứ đã thay đổi. Bữa ăn đầu tiên của nàng dâu mới chỉ có độc một lùm rau chuối thái rối và một đĩa muối ăn với cháo, nhưng họ vẫn vui vẻ vừa ăn vừa nói chuyện đến tương lai. Bà cụ Tứ bê nồi cháo cám lên, người vợ nhặt vẫn điềm nhiên và vào miệng. Tràng cầm đôi đũa gạt một miếng bỏ vội vào miệng. Nghe tiếng trống thúc thuế Tràng nhớ đến cảnh người ta vào kho thóc chia cho người đói và hình ảnh lá cờ đỏ thắm.
Truyện ngắn Vợ nhặt đã miêu tả tình cảnh thê thảm của người nông dân ta trong nạn đói khủng khiếp năm 1945. Đồng thời tác giả có thể hiện được bản chất tốt đẹp và sức sống kỳ diệu của họ. Với tình huống truyện độc đáo, cách kể chuyện hấp dẫn khắc họa khung cảnh chân thực như cảnh người chết đói, cảnh bữa cơm ngày đói với nhiều chi tiết đắt giá giọt nước mắt của bà cụ Tứ, nồi cháo cám.... Kim Lân đã miêu tả tâm lý nhân vật tinh tế với ngôn ngữ phù hợp.
Ý nghĩa nhan đề Vợ Nhặt trước hết từ vợ là một danh từ thiêng liêng dùng để chỉ người phụ nữ trong mối quan hệ được pháp luật công nhận với chồng. Theo phong tục vợ chồng chỉ được công nhận khi có sự chứng kiến của họ hàng làng xóm. Còn nhặt là hành động cầm vật bị đánh rơi lên. Kim Lân đã sáng tạo muốn nhan đề độc đáo vì người ta chỉ nói nhặt được một món đồ nào đó chứ không ai nhặt được một con người về làm vợ bao giờ cả. Nhưng qua đó nhà văn đã thể hiện được cảnh ngộ của con người lúc bấy giờ. Với nhan đề Vợ Nhặt trước hết khái quát được tình huống của truyện, đồng thời nó cũng là lời kết án đáng tiếc của Kim Lân đối với chế độ thực dân đã đẩy người nông dân vào tình cảnh nghèo đói người chết. Nhan đề Vợ Nhặt có tính khái quát cao hoàn cảnh của chàng chỉ là một trong số đó. Đồng thời qua nhan đề nhà văn cũng thể hiện sự đồng cảm xót xa cho cảnh ngộ của người nông dân trong đoạn đói năm 1945.
'''

from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

summary_text = '''
 đến khi bà cụ tứ trở về vô cùng ngạc nhiên khi thấy có người phụ nữ lạ trong nhà, nghe con kể rõ sự tình người mẹ nghèo khổ ấy đã hiểu ra và  chấp nhận nàng dâu mới động viên các con cố gắng làm ăn. . nghe tiếng trống thúc thuế tràng nhớ đến cảnh người ta vào kho thóc chia cho người đói và hình ảnh lá cờ đỏ thắm.. truyện ngắn vợ nhặt đã miêu tả tình cảnh thê thảm của người nông dân ta trong nạn đói khủng khiếp năm 1945. đồng thời tác giả có thể hiện được bản chất tốt đẹp và sức sống kỳ diệu của họ. tràng là một người dân sống ngụ cư sống cùng với mẹ già. với tình huống truyện độc đáo, cách kể chuyện hấp dẫn khắc họa khung cảnh chân thực như cảnh người chết đói, cảnh bữa cơm ngày đói với nhiều chi tiết đắt giá giọt nước mắt của bà cụ tứ, nồi cháo cám.... kim lân đã miêu tả tâm lý nhân vật tinh tế với ngôn ngữ phù hợp.. ý nghĩa nhan đề vợ nhặt trước hết từ vợ là một danh từ thiêng liêng dùng để chỉ người phụ nữ trong mối quan hệ được pháp luật công nhận với chồng. một lần trên đường kéo cái xe bò thóc trên tỉnh anh quen được thị, chỉ với bốn bát bánh đúc thị đã đồng ý làm vợ tràng. đồng thời qua nhan đề nhà văn cũng thể hiện sự đồng cảm xót xa cho cảnh ngộ của người nông dân trong đoạn đói năm 1945..
'''

# ==== TIỀN XỬ LÝ ====
nltk.download('punkt')

original_sentences = sent_tokenize(original_text.strip())
summary_sentences = sent_tokenize(summary_text.strip())

# ==== HÀM NHÚNG CÂU ====
def sentence_embedding_w2v(sentence, model):
    words = word_tokenize(sentence.lower())
    word_vectors = []
    for word in words:
        if word in model:
            word_vectors.append(model[word])
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# ==== NHÚNG CÂU ====
original_embeddings = np.array([sentence_embedding_w2v(s, w2v_model) for s in original_sentences])
summary_embeddings = np.array([sentence_embedding_w2v(s, w2v_model) for s in summary_sentences])

# ==== SO SÁNH SỐ LƯỢNG CÂU SO VỚI VĂN BẢN GỐC ====
num1 = len(original_sentences)
num2 = len(summary_sentences)
print(f"Số câu trong văn bản gốc: {num1}")
print(f"Số câu trong văn bản tóm tắt: {num2}")
if (num1 > num2):
    print(f"Văn bản tóm tắt ngắn hơn văn bản gốc {num1 - num2} câu.")

# ==== TÍNH CHÍNH XÁC NGỮ NGHĨA ====
faithful_matches = 0
for summary_vec in summary_embeddings:
    sims = cosine_similarity([summary_vec], original_embeddings)
    if np.max(sims) > 0.7:
        faithful_matches += 1

faithfulness_score = faithful_matches / len(summary_sentences)

# ==== KẾT QUẢ ====
print(f"Tính chính xác ngữ nghĩa: {faithfulness_score:.2f}")
