import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

try:
    from chat_service import get_ai_response
except Exception:
    def get_ai_response(msg, ctx=None):
        return "Layanan AI Agronomi sedang memuat. Silakan ajukan pertanyaan kembali!"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)
app.secret_key = 'tomatovision_yolo_pure_master_v53'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

CAMERA_SOURCE = 0
CAMERA_ACTIVE = False

LAST_SCAN_CONTEXT = {
    'has_scan': False,
    'total': 0,
    'matang': 0,
    'setengah': 0,
    'mentah': 0,
    'timestamp': None
}

# ==============================================================================
# 1. INISIALISASI MODEL YOLO (TOMATO DETECTOR + GUARD MODEL)
# ==============================================================================
MODEL_PATH = os.path.join('model', 'tomato_detector.pt')
if not os.path.exists(MODEL_PATH):
    fallback_path = os.path.join('model', 'best.pt')
    MODEL_PATH = fallback_path if os.path.exists(fallback_path) else 'best.pt'

tomato_detector = YOLO(MODEL_PATH)
person_guard = YOLO('yolov8n.pt')

print("=" * 60)
print(f"[TomatoVision AI] Model Tomat Aktif : {MODEL_PATH}")
print(f"[TomatoVision AI] Kelas Terdaftar   : {tomato_detector.names}")
print(f"[TomatoVision AI] Guard Filter      : Aktif (Manusia, Gelas, Cangkir, Botol)")
print("=" * 60)


# ==============================================================================
# 2. PEMETAAN LABEL KEMATANGAN TOMAT
# ==============================================================================
def map_yolo_label(raw_name):
    name = str(raw_name).lower().replace(' ', '_').replace('-', '_').strip()

    # 1. Setengah Matang ('half_ripe', 'turning')
    if 'half' in name or 'setengah' in name or 'turning' in name or 'orange' in name or 'yellow' in name:
        return 'Setengah Matang', 'setengah', (0, 215, 255)

    # 2. Muda / Mentah ('not_ripe', 'raw', 'unripe')
    elif 'not_ripe' in name or 'raw' in name or 'unripe' in name or 'mentah' in name or 'muda' in name or 'green' in name:
        return 'Muda/Mentah', 'mentah', (39, 110, 243)

    # 3. Matang Penuh ('ripe')
    elif 'ripe' in name or 'matang' in name or 'red' in name:
        return 'Matang', 'matang', (46, 204, 113)

    return 'Matang', 'matang', (46, 204, 113)


# ==============================================================================
# 3. PIPELINE DETEKSI CITRA DENGAN FILTER MULTI-OBJEK
# ==============================================================================
def process_frame(img, is_live=False):
    global LAST_SCAN_CONTEXT
    orig_h, orig_w = img.shape[:2]
    total_frame_area = float(orig_h * orig_w)

    # Guard Model: Manusia (0), Botol (39), Gelas Kaki (40), Cangkir/Mug (41)
    guard_results = person_guard.predict(
        source=img,
        classes=[0, 39, 40, 41],
        conf=0.25,
        device='cpu',
        verbose=False
    )
    guard_boxes = []
    if len(guard_results) > 0 and guard_results[0].boxes is not None:
        guard_boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in guard_results[0].boxes]

    # Deteksi Tomat dengan YOLO kustom
    conf_thresh = 0.45 if is_live else 0.35
    results = tomato_detector.predict(
        source=img,
        conf=conf_thresh,
        iou=0.40,
        device='cpu',
        verbose=False
    )

    detected_objects = []
    c_matang, c_setengah, c_mentah = 0, 0, 0

    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf_val = float(box.conf[0].cpu().numpy()) * 100.0
            cls_id = int(box.cls[0].cpu().numpy())
            raw_label = tomato_detector.names.get(cls_id, str(cls_id))

            bw = x2 - x1
            bh = y2 - y1
            box_area = float(bw * bh)

            # Filter 1: Objek terlalu besar (latar belakang / wajah dekat)
            if box_area > (total_frame_area * 0.35):
                continue

            # Filter 2: Bentuk tidak proporsional untuk buah bulat/oval
            aspect = float(bw) / float(bh + 1e-5)
            if not (0.55 <= aspect <= 1.75):
                continue

            # Filter 3: Tolak jika berada di area manusia, gelas, atau botol
            is_blocked = False
            for gx1, gy1, gx2, gy2 in guard_boxes:
                ix1, iy1 = max(x1, gx1), max(y1, gy1)
                ix2, iy2 = min(x2, gx2), min(y2, gy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = float((ix2 - ix1) * (iy2 - iy1))
                    if (inter_area / box_area) > 0.15:
                        is_blocked = True
                        break
            if is_blocked:
                continue

            final_label, badge_type, box_color = map_yolo_label(raw_label)

            if final_label == 'Matang':
                c_matang += 1
            elif final_label == 'Setengah Matang':
                c_setengah += 1
            else:
                c_mentah += 1

            # Gambar Bounding Box
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2 if is_live else 3)

            # Gambar Badge Tag
            text = f"{final_label}: {conf_val:.0f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            tag_y1 = max(0, y1 - 20)
            tag_y2 = max(20, y1)
            cv2.rectangle(img, (x1, tag_y1), (x1 + tw + 6, tag_y2), box_color, -1)
            cv2.putText(img, text, (x1 + 3, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

            detected_objects.append({
                'label': final_label,
                'confidence': f"{conf_val:.1f}%",
                'type': badge_type
            })

    # Perbarui data scan untuk Chat AI
    LAST_SCAN_CONTEXT = {
        'has_scan': True,
        'total': len(detected_objects),
        'matang': c_matang,
        'setengah': c_setengah,
        'mentah': c_mentah,
        'timestamp': time.strftime("%H:%M:%S")
    }

    if is_live:
        hud_bg = img.copy()
        cv2.rectangle(hud_bg, (10, 10), (orig_w - 10, 48), (20, 24, 33), -1)
        cv2.addWeighted(hud_bg, 0.75, img, 0.25, 0, img)
        hud_text = f"LIVE YOLO | Total: {len(detected_objects)} | Matang: {c_matang} | Setengah: {c_setengah} | Mentah: {c_mentah}"
        cv2.putText(img, hud_text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    return img, detected_objects


# ==============================================================================
# 4. ROUTE CHAT AI & FLASK
# ==============================================================================
@app.route('/chat_ai', methods=['POST'])
def chat_ai():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'reply': 'Silakan ajukan pertanyaan seputar buah tomat, pemupukan, hama, atau hasil panen!'})

    reply = get_ai_response(user_message, LAST_SCAN_CONTEXT)
    return jsonify({
        'reply': reply,
        'scan_context': LAST_SCAN_CONTEXT
    })


def create_error_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 35, 45)
    cv2.putText(frame, "KAMERA TIDAK TERHUBUNG", (135, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 76, 231), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (75, 255),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 205, 215), 1, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def generate_frames():
    global CAMERA_SOURCE, CAMERA_ACTIVE
    CAMERA_ACTIVE = True

    if isinstance(CAMERA_SOURCE, int):
        camera = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(CAMERA_SOURCE)
    else:
        camera = cv2.VideoCapture(CAMERA_SOURCE)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        err_bytes = create_error_frame(f"Gagal membuka sumber kamera: {CAMERA_SOURCE}.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + err_bytes + b'\r\n')
        return

    current_source = CAMERA_SOURCE
    try:
        while CAMERA_ACTIVE:
            if CAMERA_SOURCE != current_source:
                break

            success, frame = camera.read()
            if not success:
                err_bytes = create_error_frame("Sinyal video terputus.")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + err_bytes + b'\r\n')
                break

            if CAMERA_SOURCE == 0:
                frame = cv2.flip(frame, 1)
            else:
                frame = cv2.resize(frame, (640, 480))

            processed_frame, _ = process_frame(frame, is_live=True)

            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)
    finally:
        camera.release()


@app.route('/')
def index():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = False
    return render_template('index.html', mode='upload', current_cam=CAMERA_SOURCE, img_path=None, detections=None)


@app.route('/live_view')
def live_view():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = True
    return render_template('index.html', mode='live', current_cam=CAMERA_SOURCE)


@app.route('/set_camera', methods=['POST'])
def set_camera():
    global CAMERA_SOURCE, CAMERA_ACTIVE
    cam_input = request.form.get('cam_source', '0').strip()

    if cam_input.isdigit():
        CAMERA_SOURCE = int(cam_input)
    else:
        if cam_input.startswith('http') and not cam_input.endswith('/video'):
            cam_input = cam_input.rstrip('/') + '/video'
        CAMERA_SOURCE = cam_input

    CAMERA_ACTIVE = True
    return redirect(url_for('live_view'))


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict', methods=['POST'])
def predict():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = False

    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']
    filename = f"{int(time.time())}_{secure_filename(file.filename) or 'tomato.jpg'}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    img = cv2.imread(input_path)
    if img is None:
        return redirect(url_for('index'))

    h, w = img.shape[:2]
    if max(h, w) > 960:
        scale = 960 / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    output_filename = f"result_{filename}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)
    hist_filename = f"hist_{filename}.png"
    hist_path = os.path.join(RESULT_FOLDER, hist_filename)

    processed_img, results = process_frame(img, is_live=False)
    cv2.imwrite(output_path, processed_img)

    plt.figure(figsize=(6, 3.8), facecolor='#ffffff')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for col, lab, c_hex in zip([0, 1, 2], ['Red', 'Green', 'Blue'], ['#e53935', '#43a047', '#1e88e5']):
        hist = cv2.calcHist([img_rgb], [col], None, [256], [0, 256])
        plt.plot(hist / hist.max() if hist.max() > 0 else hist, color=c_hex, label=f'{lab} Channel', linewidth=2)
    plt.title('Sebaran Intensitas Warna RGB', fontsize=11, fontweight='bold')
    plt.xlabel('Piksel')
    plt.ylabel('Frekuensi')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=130)
    plt.close('all')

    stats = {
        'total': len(results),
        'matang': sum(1 for d in results if d['label'] == 'Matang'),
        'setengah': sum(1 for d in results if d['label'] == 'Setengah Matang'),
        'mentah': sum(1 for d in results if d['label'] == 'Muda/Mentah')
    }

    return render_template('index.html',
                           mode='upload',
                           current_cam=CAMERA_SOURCE,
                           img_path=os.path.join('results', output_filename).replace('\\', '/'),
                           hist_path=os.path.join('results', hist_filename).replace('\\', '/'),
                           detections=results,
                           stats=stats,
                           download_filename=output_filename)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)


@app.route('/reset')
def reset():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = False
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
    return redirect(url_for('index'))


if __name__ == '__main__':
    print("==================================================")
    print(" TomatoVision AI Server Aktif (Guarded YOLO Engine)!")
    print(" Buka browser di: http://127.0.0.1:5000")
    print("==================================================")
    app.run(debug=True, threaded=True, use_reloader=False)