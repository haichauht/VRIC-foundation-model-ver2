# Metrics và lý do chọn theo từng Pretraining Task

## 1. Masked Reconstruction

### Metrics

- **Primary:** Cross-Entropy Loss trên token bị mask.
- **Secondary:** Top-1 Accuracy, Perplexity (PPL), Recall@K.
- Với giá trị số: sử dụng MSE/RMSE.

### Lý do chọn

- **Phù hợp với pretraining:** Masked reconstruction buộc mô hình học hiểu ngữ cảnh toàn bộ chuỗi hành vi của user để dự đoán phần bị ẩn. Cross-Entropy và Accuracy trực tiếp đo khả năng học ngữ cảnh này.
- **Liên kết với Food Recommendation:** Nếu mô hình có thể tái tạo tốt event bị mask, có nghĩa là embedding của user đã chứa thông tin về chuỗi hành vi. Điều này giúp downstream ranking như Food
  Recommendation có thông tin user đầy đủ hơn, cải thiện NDCG@10 và Recall@20. Ví dụ, nếu sequence liên quan đến món ăn, tốt hơn trong mask nghĩa là nhận diện được sở thích lâu dài, giúp tăng khả năng lấy đúng nhà hàng/food trong top đề xuất.
- **Liên kết với Ad Recommendation:** Mặc dù ad ranking cũng dùng CTR metrics, khả năng nắm bắt ngữ cảnh qua masked có thể tăng cường tính cá nhân hóa của embedding, hỗ trợ cải thiện GAUC/LogLoss khi embedding dùng làm input cho dự đoán click.
- **Phù hợp với nhiều loại dữ liệu:** Cross-Entropy phù hợp với categorical, ID và token; MSE/RMSE phù hợp với các giá trị số bị mask.

### Quy trình đánh giá đề xuất

* Chia dữ liệu theo thời gian: Dữ liệu hành vi user được chia thành training/validation/test theo mốc thời gian (trong quá khứ huấn luyện, trong khoảng tiếp theo validation, tương lai test).
  Không random split, để tránh “nhìn trước” tương lai.
* Negative sampling: Tất cả experiment sử dụng cùng tỉ lệ negative sampling cho masked tasks(ví dụ chọn thêm tokens giả làm negative với tỉ lệ nhất định) giống hệt nhau, để so sánh công bằng.
* Tập ứng viên (candidate set): Đối với kết quả masked prediction, không cần chọn ứng viên vì nó là classification từ toàn bộ vocabulary. Đối với gợi ý (nếu có ranking top-K), ứng viên có thể là toàn bộ tập ID hoặc tập con. Đảm bảo mọi model tiền huấn luyện đánh giá trên cùng không gian token.
* Seed: Chạy mỗi experiment với ít nhất 3 seed ngẫu nhiên khác nhau, báo cáo giá trị trung bình và độ lệch chuẩn (mean±std)
* Checkpoint: Chọn checkpoint dựa trên performance (loss hoặc accuracy) trên tập validation masked (tức early stopping). Tuy nhiên, tập test không được chạm vào trong quá trình tuning.
* Kiểm tra độ ổn định: Có thể dùng bootstrap hoặc t-test (paired) trên các lần chạy seed để kiểm định sự khác biệt có ý nghĩa thống kê giữa các phiên bản.

## 2. Predict-Next-Token

### Metrics

- **Primary:** Cross-Entropy Loss cho dự đoán event kế tiếp.
- **Secondary:** Top-1 Accuracy, Recall@K, MRR.
- Nếu dự đoán giá trị số: sử dụng MSE.

### Lý do chọn

- **Phù hợp với pretraining:** Predict-Next-Token đánh giá trực tiếp khả năng dự đoán event kế tiếp của user, bao gồm type và value. Cross-Entropy là metric phù hợp cho bài toán dự đoán bước tiếp theo.
- **Liên kết với Food Recommendation:** Dự đoán chính xác hành động hoặc món ăn tiếp theo giúp embedding thể hiện sở thích và hành vi của user, hỗ trợ cải thiện NDCG@10 và Recall@20.
- **Liên kết với Ad Recommendation:** Dự đoán event kế tiếp giúp mô hình nắm bắt ý định ngắn hạn của user, có thể hỗ trợ tăng GAUC và giảm LogLoss.
- **Đánh giá thứ hạng dự đoán:** Accuracy đo tỷ lệ dự đoán đúng; Recall@K đo event thực tế có xuất hiện trong top-K hay không; MRR đánh giá thứ hạng của event thực tế trong danh sách dự đoán.

### Quy trình đánh giá đề xuất

* Chia dữ liệu theo thời gian: Sử dụng time-based split. Tập train chứa các chuỗi trước thời điểm T, test bao gồm các chuỗi sau T.
* Candidate generation: Các sự kiện tiếp theo có thể là phần tử từ toàn bộ vocabulary (rất lớn). Nếu đánh giá top-K, cần xác định tập ứng viên (có thể là toàn bộ ID, hoặc một tập gồm ID đã xuất hiện). Đảm bảo cùng một tập ứng viên cho mọi phương án (B1, B2, B3, B4) theo baseline
* Negative sampling: Với next-token, có thể không cần negative sampling riêng nếu đánh giá trực tiếp Accuracy hoặc MRR (vì chúng đã tính trên vocabulary). Nếu dùng metrics như Recall@K thường lấy K nhỏ nên cần xác định top-K bằng beam search hoặc tính softmax trên toàn vocabulary (không sampling). Dù sao, mọi experiment phải nhất quán về cách lấy top-K.
* Seed và checkpoint: Tương tự phần 1. Seed ≥3, báo cáo trung bình±độ lệch. Checkpoint chọn theo loss hoặc metric trên tập validation next-token (ví dụ đo Accuracy hoặc Recall trên validation để early stop).
