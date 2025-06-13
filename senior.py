from flask import Flask, Response, request, jsonify
from flask_cors import CORS  # ✅ لإتاحة الوصول من تطبيقات الويب
import cv2
import requests
import threading
import time
import os
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import math

MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    import gdown
    gdown.download("https://drive.google.com/uc?export=download&id=1MhhJANrwUMM3gCa4Si0kp7n0yhrLwDM7", MODEL_PATH, quiet=False)

app = Flask(__name__)
CORS(app)  # ✅ تفعيل CORS لتجنب مشاكل الوصول من المتصفح أو Flutter Web

model = YOLO(MODEL_PATH)

camera_url = None
CAMERA_URL_FILE = "camera_url.txt"

backend_url = "http://farmsmanagement.runasp.net/api/Notifiactions/CreateNotification"
user_id = 24
barn_id = 3

ADVICE_DB = {
    "sick": {
        "low": "الفرخة باين عليها تعب خفيف، حاول توفرلها مكان هادي ومراقبة لمدة ساعة. تأكد من التهوية والأكل.",
        "medium": "الفرخة مش في حالتها الطبيعية، اعزلها مؤقتًا وراقب هل بتتحرك أو تاكل. خليك مركز معاها خلال اليوم.",
        "high": "الفرخة أعراضها واضحة، اعزلها فورًا، وفر لها تهوية وأكل كويس، ولو استمر الوضع كده، استشر طبيب بيطري."
    },
    "dead": {
        "low": "الفرخة مش بتتحرك كويس، لكن لسه بدري نحكم. جرب تحفزها أو تتابعها شوية.",
        "medium": "في علامات قوية إنها مش حية، افصلها مؤقتًا عن باقي الفراخ وتابع أي تغير.",
        "high": "الفرخة غالبًا ميتة، اتصرف بسرعة بعزلها وتخلص منها بطريقة صحية. نظف مكانها وتابع باقي الفراخ."
    }
}

def get_confidence_level(conf):
    if conf >= 0.9:
        return "high"
    elif conf >= 0.7:
        return "medium"
    else:
        return "low"

def get_advice(label, conf):
    level = get_confidence_level(conf)
    return ADVICE_DB.get(label, {}).get(level, "تابع الفرخة وراقب سلوكها.")

def send_notification(body_text):
    data = {
        "body": body_text,
        "userId": user_id,
        "barnId": barn_id,
        "isRead": False
    }
    try:
        res = requests.post(backend_url, json=data)
        print("✅ Notification sent:", res.status_code, res.text)
    except Exception as e:
        print("❌ Notification error:", e)

recent_detections = deque(maxlen=50)
COOLDOWN_SECONDS = 30

def is_duplicate_detection(x, y, label):
    now = datetime.now()
    for item in recent_detections:
        prev_x, prev_y, prev_label, timestamp = item
        dist = math.hypot(x - prev_x, y - prev_y)
        if dist < 50 and label == prev_label and (now - timestamp).total_seconds() < COOLDOWN_SECONDS:
            return True
    return False

def monitor_camera():
    global camera_url
    if not camera_url:
        print("❌ No camera URL set.")
        return

    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("❌ Camera not found or not opened")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        results = model(frame)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                label = model.names[cls].lower()
                conf = float(box.conf[0])

                if label in ["sick", "dead"]:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    if not is_duplicate_detection(center_x, center_y, label):
                        advice = get_advice(label, conf)
                        message = f"🚨 تم اكتشاف حالة: {label} (نسبة الثقة: {conf:.2f})\n📝 النصيحة: {advice}"
                        send_notification(message)
                        recent_detections.append((center_x, center_y, label, datetime.now()))
        time.sleep(3)

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global camera_url
    data = request.get_json()
    camera_url = data.get("camera_url")

    if not camera_url:
        return jsonify({"success": False, "message": "❌ ابعت لينك الكاميرا"}), 400

    with open(CAMERA_URL_FILE, "w") as f:
        f.write(camera_url)

    threading.Thread(target=monitor_camera).start()
    return jsonify({"success": True, "message": f"✅ بدأنا نتابع الكاميرا: {camera_url}"}), 200

@app.route('/')
def index():
    return "🐔 Chicken Monitor is running"

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global camera_url
        if not camera_url:
            return

        cap = cv2.VideoCapture(camera_url)
        while True:
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def load_camera_url():
    global camera_url
    if os.path.exists(CAMERA_URL_FILE):
        with open(CAMERA_URL_FILE, "r") as f:
            camera_url = f.read().strip()
        threading.Thread(target=monitor_camera).start()

if __name__ == '__main__':
    load_camera_url()
    app.run(host='0.0.0.0', port=5000)
