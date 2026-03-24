import json
from vncv.ocr import extract_text

def test_ocr():
    # Đường dẫn ảnh ví dụ (thay thế bằng ảnh thật của bạn)
    image_path = r"C:\workspace\New folder\VNCV\images\raw\image.png"
    
    print(f"--- Đang chạy OCR Tiếng Việt ---")
    # Gọi hàm trích xuất text với lang='vi' (Tiếng Việt) và return_dict=True
    results_vi = extract_text(
        filepath=image_path,
        lang="vi",
        return_dict=True
    )
    
    # `results_vi` là một tuple/list các Dictionary
    for idx, row in enumerate(results_vi):
        print(f"[{idx}] Text: {row['text']} | Conf: {row['confidence']:.2f}")

    print("\n--- Chuỗi JSON hoàn chỉnh ---")
    # Convert sang định dạng JSON string đẹp mắt nếu cần API
    json_output = json.dumps(results_vi, ensure_ascii=False, indent=2)
    print(json_output)

if __name__ == "__main__":
    test_ocr()
