<div align="center">

# 🇻🇳 VNCV — Vietnamese Computer Vision

**OCR engine tối ưu hoá cho tiếng Việt**  
`vncv` là thư viện OCR hỗ trợ trích xuất văn bản từ ảnh, với khả năng tự động tải model và chạy inference nhanh chóng.

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Open%20Source-22c55e?style=flat-square)](LICENSE)
[![DevHub](https://img.shields.io/badge/By-DevHub%20Solutions-6366f1?style=flat-square)](https://github.com)
[![VietOCR](https://img.shields.io/badge/Powered%20by-VietOCR%20ONNX-f97316?style=flat-square&logo=onnx&logoColor=white)](https://github.com/pbcquoc/vietocr)

</div>

---

## 🖼️ Demo

| 📥 Ảnh gốc | 📤 Kết quả nhận dạng |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/Devhub-Solutions/VNCV/main/images/raw/image.png" width="380"/> | <img src="https://raw.githubusercontent.com/Devhub-Solutions/VNCV/main/images/output/image.png" width="380"/> |
---
```
['UBND QUẬN TÂY HỒ', 'TRƯỜNG MN-TH SAO MAI', 'TUYÊN TRUYỀN', 'Phổ biến giáo dục pháp luật về phòng, chống dịch bệnh COVID-19', 'tại nhà trường', 'Thực hiện Kế hoạch số 43/KH-PGDĐT về việc thực hiện đợt cao điểm', 'tuyên truyền pháp luật về phòng chống dịch Covid-19 trên dịa bàn Thành phố', 'Ngành GDĐT Tây Hồng', 'Nhằm nâng cao ý thức tự giác cho CB-GV-NV của trường về việc chấp', 'hành các quy định của pháp luật liên quan đến phòng, chống dịch COVID-19;', 'góp phần đẩy lùi nhanh dịch bệnh trên địa bàn thành phố, thực hiện đẩy mạnh', 'đợt cao điểm tuyên truyền pháp luật về phòng, chống dịch COVID-19 theo', 'hướng lựa chọn nội dung trọng tâm, trọng điểm, ngắn gọn, dễ hiểu, đa dạng hóa', 'các hình thức liên quan đến phòng, chống dịch, góp phần hình thành thói quen', 'thực hiện các biện pháp phòng, chống dịch trong lối sống.', 'Các quy định của pháp luật có liên quan đến phòng, chống dịch COVID-19, các', 'văn bản chỉ đạo của Thành phố về phòng, chống dịch, tình hình dịch bệnh tại', 'xã thành phố, các quy định người dân cân tuân thủ khi chính quyền áp dụng biện thuận', 'xã pháp phòng, chống dịch tại địa bàn, đặc biệt là thời gian áp dụng các biện pháp', 'theo Chỉ thị 15/CT-TTg, Chỉ thị 16/CT-TTg của Thủ tướng Chính phủ hoặc các', 'biện pháp cao hơn.', 'Xử phạt các hành vi, vi phạm pháp luật có liên quan đến phòng, chống', 'dịch, quy định về: cách ly y tế, chữa bệnh, quy định tiêm chủng vaccine của', 'thành phố... đã được triển khai mạnh trong thời gian tới.', 'Thực hiện tuyên truyền trên phân mêm ứng dụng internet: Zalo, website..', 'Trường MN-TH Sao Mai yêu cầu toàn bộ CB-GV-NV-HS thực hiện đợt cao', 'điểm tuyên truyền tại nhà trường, phối hợp và triển khai thực hiện đảm bảo đúng', 'tiến độ và hiệu quả./.', 'TRƯỞNG BANH', 'Nguyễn/Thị Trà Giang']
```
<img src="https://raw.githubusercontent.com/Devhub-Solutions/VNCV/main/images/logs_image/image.png"/>

## 1. Cài đặt

Cài đặt package thông qua pip:

```bash
pip install vncv
```

Hoặc nếu bạn muốn sử dụng GPU (tăng tốc tối đa):

```bash
pip install vncv[gpu]
```

**🚀 Điểm mới:** `vncv` hiện đã chuyển hoàn toàn sang **ONNX Runtime Engine**. Bạn không còn cần cài đặt `PyTorch` (nặng ~2GB) nữa. Thư viện giờ đây siêu nhẹ, cài đặt nhanh và chạy mượt mà ngay cả trên CPU yếu.

Trong quá trình build hoặc cài đặt lần đầu, các mô hình OCR cần thiết sẽ được **tự động tải về** và cấu hình sẵn.

---

## 2. Sử dụng cơ bản

### Trích xuất văn bản từ ảnh

```python
from vncv.ocr import extract_text

# Trích xuất văn bản từ ảnh tiếng Việt (mặc định)
results = extract_text("test_image.jpg", lang="vi")

print("OCR Results:", results)
```

👉 Thư viện sẽ tự động:
* **Tự động tải weights** từ GitHub Releases nếu chưa có (không cần copy thủ công)
* Load model OCR bằng **ONNX Runtime** (tối ưu hóa tốc độ x2-x5 so với PyTorch trên CPU)
* Phát hiện vùng chứa văn bản (text detection)
* Nhận dạng nội dung (text recognition)

---

## 3. Sử dụng CLI (Command Line Interface)

Bạn có thể chạy trực tiếp từ terminal mà không cần viết code:

```bash
vncv test_image.jpg --lang vi
```

---

## 4. Các tuỳ chọn nâng cao

### 4.1. Lưu ảnh có bounding box

```bash
vncv test_image.jpg --save-annotated
```
👉 Kết quả: lưu ảnh với các khung (bounding boxes) bao quanh vùng text.

---

### 4.2. Xuất kết quả dạng JSON

```bash
vncv test_image.jpg --json
```

Hoặc sử dụng trong khối lệnh Python:

```python
from vncv.ocr import extract_text

results = extract_text("test_image.jpg", lang="vi", return_dict=True)
```

**Ví dụ output JSON:**
```json
[
  {
    "text": "Xin chao Viet Nam",
    "confidence": 0.98,
    "box": [[10, 20], [100, 20], [100, 50], [10, 50]]
  }
]
```

---

## 5. Tính năng nổi bật

* ⚡ **Tốc độ vượt trội**: Sử dụng ONNX Runtime giúp giảm latency, xử lý ảnh nhanh hơn đáng kể.
* 📦 **Siêu nhẹ**: Gỡ bỏ phụ thuộc vào PyTorch, giảm kích thước cài đặt từ GB xuống MB.
* ✅ **Tự động tải model**, không cần cấu hình phức tạp.
* ✅ **Hỗ trợ đa ngôn ngữ** (Tiếng Việt `vi` và Tiếng Anh `en`).
* ✅ **Hỗ trợ GPU/CPU**: Tự động nhận diện và tối ưu cho phần cứng hiện có.
* ✅ **Trả về JSON chi tiết** bao gồm: văn bản, bounding box, và confidence.

---

## 6. Gợi ý sử dụng

`vncv` là lựa chọn số 1 cho:
* **Server/Lambda/Edge devices**: Nơi tài nguyên (RAM/Disk) hạn chế và không có GPU.
* **Hệ thống Real-time**: Cần tốc độ phản hồi nhanh.
* **OCR tài liệu tiếng Việt**: Hóa đơn, CMND/CCCD, văn bản hành chính.
* **Tích hợp API**: Dễ dàng build các dịch vụ OCR Microservices gọn nhẹ.

---

## 7. Giấy phép & Bản quyền

**© 2026 DevHub Solutions. All rights reserved.**

Dự án được phát hành dưới **giấy phép mã nguồn mở** cho mục đích học tập và nghiên cứu. Chân thành cảm ơn tác giả dự án [VietOCR](https://github.com/pbcquoc/vietocr) đã cung cấp engine cốt lõi.

| | Quy định |
|:---:|---|
| ✅ | Sử dụng, học hỏi, chỉnh sửa mã nguồn |
| ✅ | Tích hợp vào dự án cá nhân hoặc thương mại |
| ✅ | Chia sẻ lại với điều kiện ghi nguồn |
| ❌ | Đổi tên thương hiệu hoặc nhận là sản phẩm của mình |
| ❌ | Xây dựng SaaS/dịch vụ cạnh tranh mà không có phép |
| ❌ | Xóa thông báo bản quyền này |

> **DevHub Solutions** bảo lưu quyền duy trì các phiên bản private/thương mại và có thể thay đổi điều khoản cấp phép trong các phiên bản tương lai. Để sử dụng thương mại hoặc mở rộng quyền sử dụng, vui lòng liên hệ **DevHub Solutions**.

---

<br/>
<div align="center">
Made with ❤️ by <b>DevHub Solutions</b>
</div>
