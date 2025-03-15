from flask import Flask, jsonify, request
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import requests
import json
import os
import time
import re
from urllib.parse import unquote

app = Flask(__name__)

# Thiết lập thiết bị (CPU hoặc GPU nếu có)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Thiết bị sử dụng: {device}")

# Tải mô hình CLIP
try:
    model, preprocess = clip.load("ViT-L/14", device=device)
    print("Đã tải mô hình CLIP thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình CLIP: {e}")
    raise e

# Biến toàn cục cho đặc trưng ảnh và ID
image_features = None
image_ids = None
index = None

# Hàm lấy ảnh từ API
def get_images_from_api():
    url = "http://222.252.24.198:8758/api/SanPham/ListImageSanPham"
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            print(f"Phản hồi từ API (lần thử {attempt + 1}): {json.dumps(data, indent=2)}")
            if not data.get("message", "").lower() == "success":
                return [{"status": "error", "message": f"API không thành công: {data.get('message', 'unknown error')}"}]

            images = []
            api_data = data.get("data", [])

            if not isinstance(api_data, list):
                return [{"status": "error", "message": "Trường 'data' không phải danh sách"}]

            for idx, item in enumerate(api_data):
                image_str = item.get("image") or item.get("image_base64", "")
                if not image_str:
                    print(f"Không tìm thấy trường 'image' hoặc 'image_base64' cho item: {item}")
                    continue

                item_id = item.get("id") if item.get("id") is not None else f"api_image_{idx}"
                print(f"Xử lý item (ID: {item_id}, dữ liệu: {image_str[:50]}...)")

                if image_str.startswith("data:image/png;base64,"):
                    image_base64 = image_str[len("data:image/png;base64,"):]
                    images.append({"id": item_id, "image_base64": image_base64, "type": "image/png"})
                    print(f"Đã thêm ảnh PNG (ID: {item_id})")
                elif image_str.startswith(("data:image/jpeg;base64,", "data:image/jpg;base64,")):
                    prefix = "data:image/jpeg;base64," if image_str.startswith("data:image/jpeg;base64,") else "data:image/jpg;base64,"
                    image_base64 = image_str[len(prefix):]
                    images.append({"id": item_id, "image_base64": image_base64, "type": "image/jpeg"})
                    print(f"Đã thêm ảnh JPEG (ID: {item_id})")
                else:
                    print(f"Bỏ qua mục không phải ảnh PNG hoặc JPEG (ID: {item_id}): {image_str[:50]}...")
            print(f"Tổng số ảnh PNG/JPEG tìm thấy: {len(images)}")
            return images
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                print(f"Timeout lần {attempt + 1}, thử lại sau 2 giây...")
                time.sleep(2)
                continue
            return [{"status": "error", "message": "Kết nối đến API timeout sau 10 giây sau 3 lần thử"}]
        except requests.exceptions.RequestException as e:
            return [{"status": "error", "message": f"Lỗi kết nối API: {str(e)}"}]
        except json.JSONDecodeError as e:
            return [{"status": "error", "message": f"Không thể parse JSON: {str(e)}"}]

# Hàm xử lý ảnh và tạo chỉ mục FAISS
def preprocess_images(images):
    global image_features, image_ids, index
    image_ids = []
    image_features = []
    failed_images = []
    batch_size = 32
    print(f"Tổng số mục từ API: {len(images)}")

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_images = []
        batch_ids = []
        for item in batch:
            try:
                if item["type"] not in ["image/png", "image/jpeg"]:
                    print(f"Bỏ qua mục không phải ảnh PNG hoặc JPEG (ID: {item['id']})")
                    failed_images.append(item)
                    continue

                image_base64 = item["image_base64"].strip()
                if not image_base64:
                    print(f"Chuỗi base64 rỗng (ID: {item['id']})")
                    failed_images.append(item)
                    continue

                padding_needed = (4 - len(image_base64) % 4) % 4
                image_base64 += '=' * padding_needed
                image_bytes = base64.b64decode(image_base64, validate=True)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                processed_image = preprocess(image)
                batch_images.append(processed_image)
                batch_ids.append(item["id"])
                print(f"Xử lý thành công ảnh (ID: {item['id']})")
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh từ API (ID: {item['id']}): {e}")
                failed_images.append(item)
                continue

        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_features = model.encode_image(batch_tensor).cpu().numpy()
                batch_features = batch_features.astype(np.float32)
            image_features.append(batch_features)
            image_ids.extend(batch_ids)
            del batch_tensor
            torch.cuda.empty_cache()
            print(f"Đã xử lý batch {i // batch_size + 1}: {len(batch_images)} ảnh")

    if image_features:
        image_features = np.vstack(image_features)
        image_features = np.ascontiguousarray(image_features, dtype=np.float32)
        faiss.normalize_L2(image_features)

        d = image_features.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(image_features)
        faiss.write_index(index, "image_index.faiss")
        print("Đã tạo và lưu chỉ mục FAISS tại: image_index.faiss")
    else:
        raise ValueError("Không có đặc trưng ảnh nào được trích xuất để tạo chỉ mục!")

    if not image_ids:
        raise ValueError("Không tìm thấy ảnh PNG hoặc JPEG hợp lệ từ API sau khi xử lý!")
    return image_ids

# Tải hoặc tạo chỉ mục FAISS
def create_or_load_faiss_index(index_path="image_index.faiss"):
    global index
    if os.path.exists(index_path):
        print(f"Tải chỉ mục từ: {index_path}")
        index = faiss.read_index(index_path)
    else:
        print("Không tìm thấy chỉ mục FAISS, tạo mới...")
        images = get_images_from_api()
        if "status" in images[0]:
            raise ValueError(f"Không thể lấy ảnh từ API: {images[0]['message']}")
        preprocess_images(images)

# Tìm kiếm bằng văn bản
def search_by_text(query_text, top_k=5):
    if not query_text:
        raise ValueError("Vui lòng cung cấp văn bản để tìm kiếm!")
    with torch.no_grad():
        text_input = clip.tokenize([query_text]).to(device)
        text_feature = model.encode_text(text_input).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(text_feature)
    scores, indices = index.search(text_feature, top_k)
    if indices[0][0] == -1:
        raise ValueError("Không tìm thấy ảnh tương đồng!")
    distances = 1 - scores[0]
    similar_ids = [image_ids[idx] for idx in indices[0]]
    return similar_ids, distances

# Tìm kiếm bằng hình ảnh
def search_by_image(image_base64, top_k=5):
    try:
        image_base64 = unquote(image_base64)
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        if not re.match(r'^[A-Za-z0-9+/=]+$', image_base64):
            raise ValueError("Chuỗi base64 chứa ký tự không hợp lệ.")
        
        padding_needed = (4 - len(image_base64) % 4) % 4
        image_base64 += '=' * padding_needed
        image_bytes = base64.b64decode(image_base64, validate=True)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feature = model.encode_image(processed_image).cpu().numpy()
        image_feature = np.ascontiguousarray(image_feature, dtype=np.float32)
        faiss.normalize_L2(image_feature)
        scores, indices = index.search(image_feature, top_k)
        if indices[0][0] == -1:
            raise ValueError("Không tìm thấy ảnh tương đồng!")
        distances = 1 - scores[0]
        similar_ids = [image_ids[idx] for idx in indices[0]]
        return similar_ids, distances
    except Exception as e:
        raise ValueError(f"Lỗi khi xử lý hình ảnh: {str(e)}")

# Khởi tạo chỉ mục khi ứng dụng khởi động
try:
    create_or_load_faiss_index()
except Exception as e:
    print(f"Lỗi khi khởi tạo chỉ mục FAISS: {e}")

# Các endpoint API
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is running", "message": "Use /search_text or /search_image"})

@app.route('/search_text', methods=['GET'])
def api_search_text():
    try:
        query_text = request.args.get("query_text", "")
        top_k = int(request.args.get("top_k", 5))
        if not query_text:
            return jsonify({"status": "error", "message": "query_text is required"}), 400
        similar_ids, distances = search_by_text(query_text, top_k=top_k)
        results = [{"id": id, "distance": float(dist)} for id, dist in zip(similar_ids, distances)]
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/search_image', methods=['POST'])
def api_search_image():
    print("Yêu cầu đến endpoint: /search_image")
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON body"}), 400

        image_base64 = data.get("image_base64", "")
        top_k = int(data.get("top_k", 5))

        if not image_base64:
            return jsonify({"status": "error", "message": "image_base64 is required"}), 400

        similar_ids, distances = search_by_image(image_base64, top_k=top_k)
        results = [
            {"id": id, "distance": float(dist)}
            for id, dist in zip(similar_ids, distances)
        ]
        return jsonify({"status": "success", "results": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Heroku yêu cầu dùng biến môi trường PORT
    app.run(host='0.0.0.0', port=port)