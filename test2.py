import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from gensim.models import KeyedVectors

# ==== NẠP MÔ HÌNH WORD2VEC ====
w2v_model = KeyedVectors.load_word2vec_format("vi_txt/vi.vec") # Đổi thành đường dẫn mô hình của bạn

# ==== DỮ LIỆU ĐẦU VÀO ====
# original_text = '''
# Tràng là một người dân sống ngụ cư sống cùng với mẹ già. Anh làm nghề kéo xe bò thuê. Một lần trên đường kéo cái xe bò thóc trên tỉnh anh quen được Thị, Chỉ với bốn bát bánh đúc thị đã đồng ý làm vợ Tràng. Về đến nhà Tràng phấp phỏng chờ mẹ về để thưa chuyện. Đến khi bà cụ Tứ trở về vô cùng ngạc nhiên khi thấy có người phụ nữ lạ trong nhà, nghe con kể rõ sự tình người mẹ nghèo khổ ấy đã hiểu ra và  chấp nhận nàng dâu mới động viên các con cố gắng làm ăn. Sáng hôm sau Tràng thức dậy thấy mọi thứ đã thay đổi. Bữa ăn đầu tiên của nàng dâu mới chỉ có độc một lùm rau chuối thái rối và một đĩa muối ăn với cháo, nhưng họ vẫn vui vẻ vừa ăn vừa nói chuyện đến tương lai. Bà cụ Tứ bê nồi cháo cám lên, người vợ nhặt vẫn điềm nhiên và vào miệng. Tràng cầm đôi đũa gạt một miếng bỏ vội vào miệng. Nghe tiếng trống thúc thuế Tràng nhớ đến cảnh người ta vào kho thóc chia cho người đói và hình ảnh lá cờ đỏ thắm.

# Truyện ngắn Vợ nhặt đã miêu tả tình cảnh thê thảm của người nông dân ta trong nạn đói khủng khiếp năm 1945. Đồng thời tác giả có thể hiện được bản chất tốt đẹp và sức sống kỳ diệu của họ. Với tình huống truyện độc đáo, cách kể chuyện hấp dẫn khắc họa khung cảnh chân thực như cảnh người chết đói, cảnh bữa cơm ngày đói với nhiều chi tiết đắt giá giọt nước mắt của bà cụ Tứ, nồi cháo cám.... Kim Lân đã miêu tả tâm lý nhân vật tinh tế với ngôn ngữ phù hợp.

# Ý nghĩa nhan đề Vợ Nhặt trước hết từ vợ là một danh từ thiêng liêng dùng để chỉ người phụ nữ trong mối quan hệ được pháp luật công nhận với chồng. Theo phong tục vợ chồng chỉ được công nhận khi có sự chứng kiến của họ hàng làng xóm. Còn nhặt là hành động cầm vật bị đánh rơi lên. Kim Lân đã sáng tạo muốn nhan đề độc đáo vì người ta chỉ nói nhặt được một món đồ nào đó chứ không ai nhặt được một con người về làm vợ bao giờ cả. Nhưng qua đó nhà văn đã thể hiện được cảnh ngộ của con người lúc bấy giờ. Với nhan đề Vợ Nhặt trước hết khái quát được tình huống của truyện, đồng thời nó cũng là lời kết án đáng tiếc của Kim Lân đối với chế độ thực dân đã đẩy người nông dân vào tình cảnh nghèo đói người chết. Nhan đề Vợ Nhặt có tính khái quát cao hoàn cảnh của chàng chỉ là một trong số đó. Đồng thời qua nhan đề nhà văn cũng thể hiện sự đồng cảm xót xa cho cảnh ngộ của người nông dân trong đoạn đói năm 1945.
# '''

from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)\
    
original_text = read_docx(r"D:\gitab\TT3601_NguyenThiNgocCham_BaiChinhLuan.docx")

summary_text = '''
 trong kỷ nguyên số hóa, sự bùng nổ của công nghệ thông tin và truyền thông, đặc biệt là sự lan tỏa mạnh mẽ của các ứng dụng mạng xã hội đã mang đến những tiện ích chưa từng có cho đời sống kinh tế, văn hóa, xã hội của nhân dân việt nam. tuy nhiên, chính sự phát triển vượt bậc này, với đặc trưng về tính mở, tính tương tác cao và khả năng lan truyền thông tin nhanh chóng, lại vô tình tạo ra những "kẽ hở" nguy hiểm, trở thành nơi để các thế lực thù địch lợi dụng, thực hiện các hoạt động chống phá đảng cộng sản việt nam.. sự tinh vi và nguy hiểm của các thế lực phản động và thù địch ngày càng lớn. điều đó được thể hiện thông qua các thủ đoạn, phương thức chống phá khác nhau. hơn thế nữa, chúng tập trung kích động các vấn đề nhạy cảm trong xã hội, cố tình tạo ra mâu thuẫn nội bộ, gây chia rẽ khối đại đoàn kết dân tộc, hoặc thấm chí là kích động các hành vi vi phạm pháp luật. và yêu cầu đặt ra là sự nhận diện rõ ràng bản chất và mức độ nguy hiểm, cũng như các giải pháp hiệu quả để bảo vệ vững chắc nền tảng và tư tưởng của đảng là vô cùng cấp thiết. tính đến đầu năm 2025, số lượng người dùng mạng xã hội tại việt nam đã đạt 75,2% tổng dân số, tương đương khoảng 75 triệu người dùng, tăng từ 72,7 triệu người vào năm 2024 (trích nguồn: https://dichvuseohot.com/thong-ke-ve-nguoi-dung-mang-xa-hoi-tai-viet-nam-nam-2025/).. sự phát triển của công nghệ số và truyền thông, của các nền tảng mạng xã hội mở bao gồm nhiều nhóm người dùng ở các độ tuổi khác nhau khiến cho các thủ đoạn của các thế lực thù địch đang dần tiếp cận đến bộ phận của giới trẻ. theo thống kê vietnam digital report 2023, top 5 trang mạng xã hội được sử dụng nhiều nhất tại việt nam hiện nay bao gồm: facebook (91.6%), zalo (90.1%), tiktok (77.5%), facebook messenger (77%), instagram (55.4%). trong đó, zalo là nền tảng duy nhất đến từ doanh nghiệp nội địa việt nam.. sự sàng lọc thông tin chưa được hiệu quả khi các thông tin sai sự thật vẫn đang tràn lan trên các phương tiện mạng xã hội thông qua nhiều hình thức như: văn bản, hình ảnh, video,…. trường đại học thăng long. sự phát triển của công nghệ và truyền thông đã tạo điều kiện thuận lợi cho sự phát triển của nước ta ở một vài khía cạnh như:. truyền thông và giáo dục: dễ dàng tiếp cận thông tin, học tập trực tuyến.. kinh tế: quảng bá các sản phẩm, dịch vụ hiệu quả.. giao tiếp và kết nối: tạo ra nhiều cơ hội giao tiếp, dễ dàng trong việc giao lưu văn hóa  với các bạn bè quốc tế.. thách thức. chúng triệt để lợi dụng tính ẩn danh và lan truyền nhanh chóng để đưa ra các thông tin sai lệch, xuyên tạc về lịch sử cũng như bóp méo tư tưởng, chủ trương đường lối của đảng. chúng ta không thể phủ định rằng sự phát triển của công nghệ và truyền thông đã mang lại giá trị lớn. phát tán các thông tin sai sự thật liên quan đến đảng, cũng như là lịch sử nước ta.. truyền bá các thông tin lừa đảo, gây hoang mang dư luận. kích động các vấn đề nhạy cảm gây mất cân bằng và gây chia rẽ khối đoàn kết của dân ta.. lan truyền các hành vi lệch chuẩn, chống phá nền tảng tư tưởng của đảng.. . trong bối cảnh đó, các thế lực thù địch đã triệt để tận dụng không gian mạng để thực hiện các hành vi chống phá của chúng. mục tiêu xuyên suốt của chúng là:. mặc dù đảng, nhà nước, chính phủ đã đưa ra các bộ luật liên quan đến thông tin, an ninh mạng, tuy nhiên vẫn chưa được triệt để khi các thông tin vẫn xuất hiện ở một vài nơi. phát triển hơn các công cụ, phần mềm giám sát, nhận diện tự động với độ chính xác cao nhằm phân tích và cảnh báo sớm các thông tin độc hại và sai lệch.. xây dựng các hệ thống bảo mật, tường lửa để bảo vệ dữ liệu và các cổng thông tin chính thống của đảng và nhà nước.. tăng cường nguồn nhân lực chất lượng chuyên trách về an ninh mạng, phòng chống tội phạm công nghệ cao.. hợp tác giao lưu quốc tế về lĩnh vực an ning mạng, an toàn thông tin để chia sẻ thông tin và kinh nghiệm để đối phó.. tuyên truyền, giáo dục về cách phòng, chống các thông tin sai lệch. sáng tạo các video, hình ảnh, postcard,… với thông điệp cụ thể, ngắn gọn.. sử dụng ngôn ngữ gần gũi, dễ hiểu, phù hợp với từng đối tượng, đặc biệt là giới trẻ.. tận dụng sức mạnh lan tỏa của mạng xã hội để truyền tải thông tin tích cực, định hướng dư luận.. tổ chức các cuộc thi, diễn đàn trực tuyến về lý luận chính trị, thu hút sự tham gia của đông đảo cán bộ, đảng viên và nhân dân.. .
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
if (num1 > num2):
    print(f"Văn bản tóm tắt ngắn hơn văn bản gốc {num1 - num2} câu.")

# ==== TÍNH CHÍNH XÁC NGỮ NGHĨA ====
faithful_matches = 0
for summary_vec in summary_embeddings:
    sims = cosine_similarity([summary_vec], original_embeddings)
    if np.max(sims) > 0.9:  # Ngưỡng có thể điều chỉnh
        faithful_matches += 1

faithfulness_score = faithful_matches / len(summary_sentences)

# ==== KẾT QUẢ ====
print(f"Tính chính xác ngữ nghĩa: {faithfulness_score:.2f}")
