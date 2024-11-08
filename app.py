from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
from PIL import Image

app = Flask(__name__)

# Load mô hình đã lưu
model = load_model("brain_tumor_detection_model.h5")

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Dự đoán kết quả
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Lấy file ảnh từ request
    file = request.files['file']
    
    # Chuyển đổi file tải lên thành đối tượng BytesIO và đọc bằng PIL
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))

    # Chỉnh lại kích thước và chuyển đổi ảnh thành mảng NumPy
    image = image.resize((224, 224))  # Đảm bảo kích thước phù hợp với mô hình của bạn
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch để mô hình có thể dự đoán

    # Dự đoán
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Trả kết quả dưới dạng JSON
    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
