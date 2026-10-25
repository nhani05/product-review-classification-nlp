# Báo cáo: Phân loại đánh giá sản phẩm Spam / Ham trên Điện máy xanh

Thư mục này chứa dàn ý, tài liệu báo cáo và file báo cáo chi tiết cho bài toán phân loại đánh giá sản phẩm Điện máy xanh.

## Cấu trúc báo cáo

1. **Giới thiệu về bài toán**
   - Đặt vấn đề và tính cấp thiết trong thương mại điện tử.
   - Mục tiêu phân loại nhị phân Spam / Ham.
   - Thách thức: Dữ liệu văn bản tiếng Việt mất cân bằng nhãn (Spam chiếm tỷ lệ nhỏ ~4.7%).
2. **Cơ sở lý thuyết**
   - Quy trình tiền xử lý văn bản tiếng Việt (Unicode NFC, rút gọn ký tự lặp, tách từ, loại bỏ từ dừng).
   - Phương pháp trích chọn đặc trưng: BoW (Bag of Words) và TF-IDF với n-gram (1, 2).
   - Thuật toán phân loại: Naive Bayes (MultinomialNB) và Support Vector Machine (LinearSVC).
   - Các chỉ số đánh giá: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, đặc biệt chú trọng F1 và Recall của lớp Spam.
3. **Thực nghiệm và kết quả**
   - Bộ dữ liệu: 10,114 đánh giá thu thập từ Điện Máy Xanh (9,636 Ham, 478 Spam).
   - Tỷ lệ phân chia: 80% tập huấn luyện (8,090 mẫu) - 20% tập kiểm tra (2,024 mẫu), áp dụng stratify.
   - Bảng so sánh kết quả 4 mô hình (xem `results/metrics.csv`).
   - Phân tích chi tiết và biểu đồ ma trận nhầm lẫn (`results/figures/`).
4. **Kết luận và hướng phát triển**
   - Nhận xét về sự phù hợp của TF-IDF và SVM / Naive Bayes trên dữ liệu văn bản tiếng Việt.
   - Đề xuất cải tiến: Word Embeddings (Word2Vec / PhoBERT), tinh chỉnh siêu tham số chuyên sâu hơn.
5. **Tài liệu tham khảo**

> Ghi chú: File báo cáo chính thức có thể xuất từ file `bao_cao.md` thành `bao_cao.pdf` hoặc dùng Microsoft Word / LaTeX theo mẫu của trường / môn học.
