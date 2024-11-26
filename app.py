from flask import Flask, request, render_template, jsonify, send_file
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import io
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# Load mô hình đã lưu
model = load_model("brain_tumor_detection_model.h5")

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Dự đoán kết quả và hiển thị ảnh
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Lấy file ảnh từ request
    file = request.files['file']
    
    # Chuyển đổi file tải lên thành đối tượng BytesIO và đọc bằng PIL
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))

    # Chỉnh lại kích thước ảnh (theo kích thước huấn luyện là 150x150)
    image = image.resize((150, 150))  # Đảm bảo kích thước đúng với mô hình huấn luyện

    # Chuyển đổi ảnh thành mảng NumPy
    image = img_to_array(image)
    
    # Chuẩn hóa ảnh giống như trong quá trình huấn luyện (1./255)
    image = image / 255.0  # Hoặc sử dụng preprocess_input nếu cần
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    
    # Dự đoán
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Gán tên loại u tương ứng với predicted_class
    tumor_types = {
        0: "U thần kinh đệm",
        1: "U màng não",
        2: "Không có khối u",
        3: "U tuyến yên"
    }
    
    # Lấy tên loại u từ lớp dự đoán
    predicted_tumor_type = tumor_types.get(predicted_class, "Không xác định")

    # Chuyển ảnh đã xử lý thành base64 để hiển thị trên trang web
    buffered = BytesIO()
    image = Image.open(io.BytesIO(img_bytes))
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Trả kết quả dưới dạng JSON
    return jsonify({
        'predicted_class': int(predicted_class),
        'tumor_type': predicted_tumor_type,
        'image': img_str
    })

if __name__ == '__main__':
    app.run(debug=True)
