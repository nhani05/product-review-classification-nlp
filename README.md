# Phân loại đánh giá sản phẩm Spam / Ham trên Điện máy xanh

## 1. Giới thiệu

Đánh giá (review) của khách hàng ảnh hưởng lớn đến quyết định mua hàng trên các sàn thương mại điện tử. Tuy nhiên, bên cạnh những đánh giá thật còn có nhiều đánh giá **spam**: quảng cáo, chèn link/số điện thoại, nội dung lặp lại, vô nghĩa hoặc không liên quan đến sản phẩm.

Dự án xây dựng hệ thống **phân loại nhị phân** các review sản phẩm trên **Điện máy xanh** thành:

| Nhãn | Ý nghĩa | Ví dụ |
|---|---|---|
| `ham` (0) | Review hợp lệ, nói về sản phẩm/trải nghiệm | "Máy lạnh chạy êm, làm mát nhanh, giao hàng đúng hẹn" |
| `spam` (1) | Review rác: quảng cáo, chèn liên hệ, lặp ký tự, không liên quan | "Mua hàng giá rẻ liên hệ 09xx xxx xxx" |

Mục tiêu: so sánh hiệu suất giữa **các phương pháp trích chọn đặc trưng** (BoW, TF-IDF, ...) và **các thuật toán phân loại** (Naive Bayes, SVM).

## 2. Quy trình thực hiện

```
Thu thập dữ liệu -> Gán nhãn -> Tiền xử lý -> Trích chọn đặc trưng -> Huấn luyện -> Đánh giá
```

1. **Thu thập dữ liệu:** thu thập review sản phẩm từ website Điện máy xanh.
2. **Gán nhãn:** gán nhãn `spam`/`ham` theo tiêu chí thống nhất (xem mục 5).
3. **Tiền xử lý:** chuẩn hóa văn bản, tách từ tiếng Việt, loại bỏ stop words.
4. **Trích chọn đặc trưng:** thử nghiệm ít nhất 2 phương pháp (BoW, TF-IDF, có thể thêm Word Embeddings).
5. **Xây dựng và huấn luyện mô hình:** Naive Bayes và SVM; chia tập huấn luyện/kiểm tra.
6. **Đánh giá:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix; so sánh các cấu hình.

## 3. Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Dữ liệu thô sau khi thu thập
│   ├── labeled/              # Dữ liệu đã gán nhãn (reviews_labeled.csv)
│   └── processed/            # Dữ liệu sau tiền xử lý
├── notebooks/
│   ├── 01_crawl_data.ipynb
│   ├── 02_eda_labeling.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_extraction.ipynb
│   └── 05_train_evaluate.ipynb
├── src/
│   ├── crawler.py            # Thu thập review
│   ├── preprocess.py         # Tiền xử lý văn bản
│   ├── features.py           # BoW, TF-IDF, Word Embeddings
│   ├── train.py              # Huấn luyện Naive Bayes, SVM
│   └── evaluate.py           # Tính chỉ số, vẽ confusion matrix
├── stopwords/
│   └── vietnamese-stopwords.txt
├── results/
│   ├── metrics.csv           # Bảng so sánh kết quả
│   └── figures/              # Confusion matrix, biểu đồ
└── report/
    └── bao_cao.pdf           # Báo cáo cuối cùng
```

> Cấu trúc trên là đề xuất, có thể điều chỉnh cho phù hợp với mã nguồn thực tế của nhóm.

## 4. Cài đặt

**Yêu cầu:** Python 3.9 trở lên.

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

Ví dụ `requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
underthesea        # hoặc pyvi để tách từ tiếng Việt
requests
beautifulsoup4
selenium           # nếu trang tải review bằng JavaScript
gensim             # nếu dùng Word Embeddings (Word2Vec)
jupyter
```

## 5. Dữ liệu

### 5.1. Nguồn dữ liệu
Review sản phẩm từ [dienmayxanh.com](https://www.dienmayxanh.com). Chỉ thu thập nội dung công khai, tốc độ thu thập vừa phải và tuân thủ điều khoản sử dụng của trang web.

### 5.2. Định dạng

File `data/labeled/reviews_labeled.csv`:

| Cột | Mô tả |
|---|---|
| `product` | Tên/danh mục sản phẩm |
| `review` | Nội dung đánh giá |
| `rating` | Số sao (nếu có) |
| `label` | `0` = ham, `1` = spam |

### 5.3. Tiêu chí gán nhãn spam

Một review được gán `spam` nếu thuộc một trong các trường hợp:

- Chứa quảng cáo, link, số điện thoại, tài khoản mạng xã hội.
- Nội dung lặp lại, ký tự vô nghĩa (ví dụ "aaaaaa", "đfgdfg").
- Không liên quan đến sản phẩm (nội dung chính trị, chửi bới không liên quan, ...).
- Quá ngắn và không mang thông tin (ví dụ ".", "ok ok ok").

Các trường hợp còn lại là `ham`. Nên có ít nhất 2 thành viên cùng gán nhãn một mẫu con để kiểm tra độ nhất quán.

### 5.4. Thống kê dữ liệu

| Thông tin | Giá trị |
|---|---|
| Tổng số review | *(điền sau khi thu thập)* |
| Số review ham | *(điền)* |
| Số review spam | *(điền)* |
| Tỷ lệ chia train/test | 80/20 *(hoặc theo nhóm chọn)* |

> Nếu dữ liệu mất cân bằng (spam ít hơn nhiều so với ham), cân nhắc dùng `stratify` khi chia tập, `class_weight='balanced'` hoặc kỹ thuật lấy mẫu lại.

## 6. Tiền xử lý

Các bước thực hiện trong `src/preprocess.py`:

1. Chuyển về chữ thường, chuẩn hóa Unicode tiếng Việt.
2. Xóa URL, số điện thoại, email, thẻ HTML, ký tự đặc biệt, emoji.
3. Rút gọn ký tự lặp (ví dụ "ngonnnnn" -> "ngon").
4. **Tách từ** tiếng Việt (ví dụ `underthesea.word_tokenize`).
5. **Loại bỏ stop words** bằng danh sách stop words tiếng Việt.

> Lưu ý: số điện thoại và link là dấu hiệu mạnh của spam. Nếu xóa ở bước tiền xử lý, nên tạo trước các đặc trưng phụ như `has_phone`, `has_url` để không mất thông tin.

## 7. Trích chọn đặc trưng

| Phương pháp | Mô tả | Cài đặt |
|---|---|---|
| **BoW** | Đếm số lần xuất hiện của từ | `CountVectorizer` |
| **TF-IDF** | Đếm có trọng số theo độ hiếm của từ | `TfidfVectorizer` |
| *(tùy chọn)* **Word Embeddings** | Vector ngữ nghĩa của từ (Word2Vec, ...) | `gensim` |

Có thể thử thêm n-gram (1,2) để bắt được cụm từ.

## 8. Mô hình

| Mô hình | Cài đặt |
|---|---|
| **Naive Bayes** | `MultinomialNB` |
| **SVM** | `LinearSVC` hoặc `SVC` |

Tinh chỉnh siêu tham số bằng `GridSearchCV` (ví dụ `alpha` của Naive Bayes, `C` của SVM).

## 9. Chạy chương trình

```bash
# 1. Thu thập dữ liệu
python src/crawler.py --output data/raw/reviews_raw.csv

# 2. Tiền xử lý
python src/preprocess.py --input data/labeled/reviews_labeled.csv \
                         --output data/processed/reviews_clean.csv

# 3. Huấn luyện và đánh giá (chọn đặc trưng và mô hình)
python src/train.py --features tfidf --model svm
python src/train.py --features bow   --model nb
```

Hoặc chạy lần lượt các notebook trong thư mục `notebooks/`.

## 10. Đánh giá

Các chỉ số sử dụng: **Accuracy, Precision, Recall, F1-Score, Confusion Matrix**.

Với bài toán spam, nên chú ý đặc biệt đến **Recall** và **F1 của lớp spam**, vì Accuracy có thể cao giả tạo khi dữ liệu mất cân bằng.

### Bảng kết quả (điền sau khi chạy thực nghiệm)

| Đặc trưng | Mô hình | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| BoW | Naive Bayes | 0.8958 | 0.9280 | 0.8958 | 0.9103 |
| BoW | SVM | 0.7777 | 0.9371 | 0.7777 | 0.8394 |
| TF-IDF | Naive Bayes | 0.9526 | 0.9434 | 0.9526 | 0.9466 |
| TF-IDF | SVM | 0.9526 | 0.9074 | 0.9526 | 0.9294 |

Confusion matrix và biểu đồ được lưu trong `results/figures/`.

## 11. Báo cáo

Báo cáo (`report/`) gồm các phần:

1. **Giới thiệu về bài toán**
2. **Cơ sở lý thuyết** (tiền xử lý, BoW/TF-IDF, Naive Bayes, SVM, các chỉ số đánh giá)
3. **Thực nghiệm và kết quả**
4. **Tài liệu tham khảo**

## 12. Thành viên nhóm

| STT | Họ và tên | MSSV | Công việc |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## 13. Tài liệu tham khảo
- Tài liệu scikit-learn: https://scikit-learn.org
- Thư viện underthesea: https://github.com/undertheseanlp/underthesea
- *(bổ sung các bài báo/tài liệu khác nhóm sử dụng)*