import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from collections import deque, Counter
import base64
import io
from PIL import Image, ImageFont, ImageDraw
import threading
import tempfile
import requests
import pygame
import arabic_reshaper
from bidi.algorithm import get_display

# --- ElevenLabs API Configuration ---
# It's highly recommended to use environment variables for sensitive keys in a real application
API_KEY = 'sk_e883cf78c9b9e06bb7236ca8ba711d0f5fe77281579f616b' # استبدل بمفتاح الـ API الخاص بك
VOICE_ID = 'wxweiHvoC2r2jFM7mS8b' # استبدل بمعرف الصوت الخاص بك
# --- End ElevenLabs API Configuration ---

# --- Text-to-Speech Function ---
# Ensure pygame is initialized for mixer
pygame.mixer.init()

def speak_elevenlabs(text):
    def thread_speak():
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": API_KEY
            }
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                }
            }

            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
                fp.write(response.content)
                fp.flush()
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(fp.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

        except Exception as e:
            print(f"حدث خطأ أثناء استخدام ElevenLabs: {e}")

    # Start speaking in a new thread to not block the main camera stream
    threading.Thread(target=thread_speak).start()
# --- End Text-to-Speech Function ---

# إعداد Flask و SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key' # يجب أن يكون هذا المفتاح سريًا في الإنتاج
socketio = SocketIO(app, cors_allowed_origins="*")

# متغيرات عامة للنموذج ومعالجة الفيديو
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
interpreter = None
hands_detector = None
camera_thread = None
camera_running = False
camera_cap = None
CAMERA_INDEX = 0  # رقم الكاميرا: 0 للكاميرا الافتراضية، 1 للثانية وهكذا

# --- Drawing Functions ---
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )

def draw_confidence_bar(image, confidence):
    bar_width = 200
    bar_height = 20
    fill_width = int(bar_width * confidence)
    cv2.rectangle(image, (10, 60), (10 + bar_width, 60 + bar_height), (255, 255, 255), 1)
    cv2.rectangle(image, (10, 60), (10 + fill_width, 60 + bar_height), (0, 255, 0), -1)
    cv2.putText(image, f'Confidence: {confidence:.2f}', (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def draw_arabic_text(image, text, position, font_size=36):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    
    font_path = os.path.join(os.path.dirname(__file__), "NotoNaskhArabic-Regular.ttf")
    if not os.path.exists(font_path):
        print(f"تحذير: لم يتم العثور على الخط {font_path}. قد لا يتم عرض النص العربي بشكل صحيح.")
        font = ImageFont.load_default() # Fallback to a default font
    else:
        font = ImageFont.truetype(font_path, font_size)

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Calculate text size using getbbox for more accurate positioning
    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = bbox[2] - bbox[0]
    # text_height = bbox[3] - bbox[1] # Keep for reference if needed

    # The test code had sentence at x=10, then adjusted based on image width.
    # We will position it 10px from the right edge for RTL text
    x_pos = image.shape[1] - text_width - 10
    y_pos = position[1] # Use the provided y-position

    draw.text((x_pos, y_pos), bidi_text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- End Drawing Functions ---

class SignLanguageInterpreter:
    def __init__(self, model_path):
        try:
            self.model = load_model(f'{model_path}.h5')
            print(f"تم تحميل النموذج: {model_path}.h5")
        except Exception as e:
            print(f"خطأ في تحميل النموذج: {e}")
            raise e
        
        try:
            with open(f'{model_path}_actions.txt', 'r', encoding='utf-8') as f:
                self.actions = [line.strip() for line in f if line.strip()]
            print(f"تم تحميل {len(self.actions)} إشارة من الملف")
        except FileNotFoundError:
            print(f"لم يتم العثور على ملف: {model_path}_actions.txt")
            raise FileNotFoundError(f"ملف الإشارات غير موجود: {model_path}_actions.txt")
        
        # تحميل معاملات التطبيع (Scaler)
        try:
            self.scaler_mean = np.load(f'{model_path}_scaler_mean.npy')
            self.scaler_scale = np.load(f'{model_path}_scaler_scale.npy')
            print("تم تحميل معاملات التطبيع")
        except FileNotFoundError:
            print("تحذير: ملفات التطبيع (scaler) غير موجودة، قد يؤثر ذلك على دقة النموذج.")
            # قيم افتراضية إذا لم يتم العثور على ملفات التطبيع
            self.scaler_mean = np.zeros(126) # 21 keypoints * 3 coords * 2 hands = 126
            self.scaler_scale = np.ones(126)
        
        self.no_frames = 30
        self.stability_window = 12
        self.confidence_threshold = 0.85 # Adjusted to match test code
        self.sequence = deque(maxlen=self.no_frames) # تسلسل الإطارات الحالية
        self.prediction_buffer = deque(maxlen=self.stability_window) # مخزن التنبؤات الأخيرة
        self.sentence = [] # الجملة المترجمة
        self.last_prediction_time = 0
        self.cooldown_period = 2.0 # فترة انتظار قبل إضافة نفس الكلمة مرة أخرى
        self.visual_flash = 0 # للمؤثر البصري عند إضافة كلمة جديدة
    
    def scale_features(self, features):
        # تطبيق التطبيع على النقاط المفتاحية
        return (features - self.scaler_mean) / self.scaler_scale
    
    def predict(self):
        if len(self.sequence) == self.no_frames:
            try:
                # تحويل التسلسل إلى مصفوفة وتطبيعها
                scaled_sequence = self.scale_features(np.array(self.sequence))
                # التنبؤ باستخدام النموذج
                res = self.model.predict(np.expand_dims(scaled_sequence, axis=0), verbose=0)[0]
                confidence = np.max(res)
                predicted_class = np.argmax(res)
                return predicted_class, confidence
            except Exception as e:
                print(f"خطأ في التنبؤ: {e}")
                return None, 0
        return None, 0
    
    def update_state(self, predicted_class, confidence):
        current_time = time.time()
        new_word_added = False
        predicted_word_text = None

        if confidence >= self.confidence_threshold:
            self.prediction_buffer.append(predicted_class)
            
            if len(self.prediction_buffer) == self.stability_window:
                # الحصول على التنبؤ الأكثر شيوعًا في المخزن المؤقت
                common = Counter(self.prediction_buffer).most_common(1)[0]
                
                # إذا كانت نسبة تكرار الكلمة الشائعة كافية (80% من نافذة الثبات)
                if common[1] >= int(0.8 * self.stability_window):
                    final_pred = common[0]
                    if final_pred < len(self.actions):
                        predicted_word_text = self.actions[final_pred]
                    else:
                        predicted_word_text = "Unknown" # حالة غير متوقعة
                    
                    # إضافة الكلمة إلى الجملة إذا كانت جديدة أو مرت فترة التهدئة
                    if (not self.sentence or # إذا كانت الجملة فارغة
                        predicted_word_text != self.sentence[-1] or # أو الكلمة مختلفة عن السابقة
                        current_time - self.last_prediction_time > self.cooldown_period): # أو مرت فترة التهدئة
                        
                        self.last_prediction_time = current_time
                        self.sentence.append(predicted_word_text)
                        self.prediction_buffer.clear() # مسح مخزن التنبؤات لبدء اكتشاف كلمة جديدة
                        self.sequence.clear() # مسح التسلسل لبدء جمع إطارات جديدة
                        self.visual_flash = 3 # لتأثير بصري
                        new_word_added = True
                        print(f"كلمة جديدة: {predicted_word_text}")
        
        return new_word_added, predicted_word_text
    
    def reset_after_no_hand(self):
        self.sequence.clear()
        self.prediction_buffer.clear()
    
    def get_current_sentence(self, max_length=10):
        # عرض آخر 10 كلمات من الجملة
        return ' '.join(self.sentence[-max_length:])
    
    def clear_sentence(self):
        self.sentence.clear()
        print("تم مسح الجملة")

def mediapipe_detection(image, model):
    # قلب الصورة أفقيًا وتحويلها إلى RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False # لجعل الصورة غير قابلة للكتابة لزيادة الأداء
    results = model.process(image) # معالجة الصورة لاكتشاف اليد
    image.flags.writeable = True # جعلها قابلة للكتابة مرة أخرى
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results # إرجاع الصورة إلى BGR

def extract_keypoints(results):
    # استخراج النقاط المفتاحية لليد اليسرى واليمنى
    lh = np.zeros(21*3) # 21 نقطة * 3 إحداثيات (x,y,z)
    rh = np.zeros(21*3)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            landmarks = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            
            # MediaPipe 'Right' hand is the user's left hand (appears on the right side of the flipped image)
            # MediaPipe 'Left' hand is the user's right hand (appears on the left side of the flipped image)
            # This mapping is crucial to match how the model was trained.
            if handedness == 'Right': # MediaPipe's 'Right' (اليد اليسرى للمستخدم)
                rh = landmarks # تخزينها في متغير اليد اليمنى
            else: # MediaPipe's 'Left' (اليد اليمنى للمستخدم)
                lh = landmarks # تخزينها في متغير اليد اليسرى
    return np.concatenate([lh, rh]) # دمج النقاط المفتاحية لليدين

# --- وظيفة تشغيل دفق الكاميرا في Thread منفصل ---
def camera_stream_thread():
    global camera_running, camera_cap, interpreter, hands_detector

    camera_cap = cv2.VideoCapture(CAMERA_INDEX)
    # ضبط حجم المخزن المؤقت إلى 1 لتقليل زمن التأخير
    camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera_cap.isOpened():
        print("خطأ: تعذر فتح الكاميرا للبث.")
        socketio.emit('camera_error', {'message': 'تعذر فتح الكاميرا. تأكد من أنها غير مستخدمة بواسطة برنامج آخر وأن الكاميرا رقم 0 صحيحة.'})
        camera_running = False
        return

    print(f"تم فتح الكاميرا رقم {CAMERA_INDEX} للبث.")
    
    last_hand_time = time.time() # لتتبع آخر مرة تم فيها اكتشاف يد

    # تهيئة MediaPipe Hands داخل الثريد
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.7
    ) as hands:
        hands_detector = hands # تعيينه للمتغير العام

        while camera_running and camera_cap.isOpened():
            success, frame = camera_cap.read()
            if not success:
                print("فشل في التقاط إطار، إعادة تشغيل الكاميرا...")
                camera_cap.release()
                time.sleep(1) # إعطاء وقت للكاميرا للتحرير
                camera_cap = cv2.VideoCapture(CAMERA_INDEX)
                camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if not camera_cap.isOpened():
                    print("فشل في إعادة فتح الكاميرا.")
                    socketio.emit('camera_error', {'message': 'تعذر إعادة فتح الكاميرا.'})
                    camera_running = False
                continue

            # معالجة الإطار باستخدام MediaPipe والنموذج
            image_processed, results = mediapipe_detection(frame, hands_detector)

            keypoints = extract_keypoints(results)
            prediction_text = ""
            confidence = 0
            new_word = False
            
            # التحقق مما إذا تم اكتشاف أي يد (نقاط مفتاحية غير صفرية)
            hand_detected = not np.all(keypoints == 0)

            if hand_detected:
                last_hand_time = time.time() # تحديث وقت آخر يد مكتشفة
                interpreter.sequence.append(keypoints)
                predicted_class, confidence = interpreter.predict()
                
                if predicted_class is not None:
                    # تحديث حالة المفسر وإضافة كلمة إذا لزم الأمر
                    new_word, predicted_text_from_update = interpreter.update_state(predicted_class, confidence)
                    
                    if new_word:
                        prediction_text = predicted_text_from_update
                        # إذا تم إضافة كلمة جديدة، يمكننا التفكير في نطقها هنا
                        # حالياً يتم النطق عند اختفاء اليد أو إيقاف البث ليتحدث الجملة كاملة
                    elif predicted_class < len(interpreter.actions):
                        # إذا لم يتم إضافة كلمة جديدة، اعرض التنبؤ الحالي الأعلى ثقة
                        prediction_text = interpreter.actions[predicted_class]
                    
            else:
                # إذا لم يتم اكتشاف يد لفترة زمنية
                if time.time() - last_hand_time > 3.0: # فترة 3 ثواني بدون يد (من كود test)
                    if interpreter.sentence: # إذا كانت هناك جملة تم تكوينها
                        full_sentence = ' '.join(interpreter.sentence)
                        print(f"نطق الجملة: {full_sentence}")
                        speak_elevenlabs(full_sentence) # نطق الجملة
                        interpreter.sentence.clear() # مسح الجملة بعد النطق
                    interpreter.reset_after_no_hand() # إعادة تعيين حالة المفسر


            # --- رسم التراكبات على الإطار (لتظهر في واجهة المستخدم) ---
            # # رسم مستطيل شفاف في الأسفل لعرض الجملة - تم إزالته للعرض في HTML
            # overlay = image_processed.copy()
            # cv2.rectangle(overlay, (0, 400), (640, 480), (0, 0, 0), -1) # أسود
            # alpha = 0.6 # شفافية
            # image_processed = cv2.addWeighted(overlay, alpha, image_processed, 1 - alpha, 0)

            # # رسم الجملة المترجمة - تم إزالته للعرض في HTML
            # sentence_display = interpreter.get_current_sentence()
            # if sentence_display:
            #     image_processed = draw_arabic_text(image_processed, sentence_display, (10, 420), 36)

            # # رسم شريط الثقة - تم إخفاؤه بناءً على طلب المستخدم
            # if confidence > 0.1: # فقط إذا كانت الثقة أعلى من حد معين للعرض
            #     draw_confidence_bar(image_processed, confidence)

            # # رسم النقاط المفتاحية - تم إخفاؤها بناءً على طلب المستخدم
            # draw_styled_landmarks(image_processed, results)

            # المؤثر البصري عند إضافة كلمة جديدة
            if interpreter.visual_flash > 0:
                cv2.rectangle(image_processed, (0, 0), (640, 480), (0, 255, 255), thickness=15) # حدود صفراء
                interpreter.visual_flash -= 1

            # ترميز الإطار إلى JPEG لإرساله عبر الويب
            ret, buffer = cv2.imencode('.jpg', image_processed, [int(cv2.IMWRITE_JPEG_QUALITY), 80]) # جودة 80%
            frame_data = base64.b64encode(buffer).decode('utf-8')

            # إرسال نتائج التنبؤ والإطار إلى المتصفح
            socketio.emit('video_frame', { # حدث جديد لإرسال الإطار
                'hand_detected': hand_detected,
                'prediction': prediction_text, # الكلمة الواحدة الحالية (ستظهر في أعلى اليمين)
                'confidence': float(confidence),
                'sentence': interpreter.get_current_sentence(), # الجملة المتراكمة (ستظهر في الأسفل)
                'new_word': new_word,
                'frame': frame_data
            })
            socketio.sleep(0.01) # تأخير بسيط لتجنب إغراق العميل والـ CPU

        # عند توقف الحلقة، تحرير الكاميرا
        camera_cap.release()
        print("تم إيقاف بث الكاميرا وتحريرها.")
        socketio.emit('camera_stopped', {'message': 'تم إنهاء بث الكاميرا.'})

# --- مسارات Flask ---
@app.route('/')
def index():
    return render_template('index.html')

# مسار للملفات الثابتة (مثل الصور ونماذج الـ GLB)
@app.route('/static/<path:filename>')
def static_files(filename):
    return app.send_static_file(filename)

# --- أحداث SocketIO ---
@socketio.on('connect')
def handle_connect():
    print('عميل متصل')
    emit('status', {'message': 'متصل بالخادم'})

@socketio.on('disconnect')
def handle_disconnect():
    print('انقطع الاتصال')
    global camera_running, camera_cap
    if camera_running:
        camera_running = False # إشارة لـ thread الكاميرا بالتوقف
        # الثريد نفسه سيحرر الكاميرا
        print("جاري إيقاف الكاميرا بسبب قطع اتصال العميل.")

@socketio.on('initialize_model')
def handle_initialize_model(data):
    global interpreter
    
    try:
        model_path = data.get('model_path', 'test') # اسم النموذج الافتراضي 'test'
        print(f"جاري تحميل النموذج: {model_path}")
        
        interpreter = SignLanguageInterpreter(model_path)
        
        emit('model_initialized', {
            'status': 'success',
            'message': 'تم تحميل النموذج بنجاح',
            'actions': interpreter.actions
        })
        print("تم تحميل النموذج بنجاح!")
        
    except Exception as e:
        error_message = f'خطأ في تحميل النموذج: {str(e)}'
        print(error_message)
        emit('model_initialized', {
            'status': 'error',
            'message': error_message
        })

@socketio.on('start_camera_stream')
def handle_start_camera_stream():
    global camera_running, camera_thread, interpreter
    if not interpreter:
        emit('camera_error', {'message': 'النموذج غير مهيأ. يرجى تحديث الصفحة والمحاولة مرة أخرى.'})
        return

    if not camera_running:
        print("بدء بث الكاميرا...")
        camera_running = True
        # بدء ثريد جديد للكاميرا
        camera_thread = threading.Thread(target=camera_stream_thread)
        camera_thread.daemon = True # يسمح للخادم بالإغلاق حتى لو كان الثريد قيد التشغيل
        camera_thread.start()
        emit('camera_stream_started', {'status': 'success', 'message': 'تم بدء بث الكاميرا.'})
    else:
        emit('camera_stream_started', {'status': 'info', 'message': 'بث الكاميرا يعمل بالفعل.'})

@socketio.on('stop_camera_stream')
def handle_stop_camera_stream():
    global camera_running, interpreter
    if camera_running:
        print("إيقاف بث الكاميرا...")
        camera_running = False # إشارة لـ thread الكاميرا بالتوقف
        # الثريد نفسه سيحرر الكاميرا
        emit('camera_stream_stopped', {'status': 'success', 'message': 'جاري إيقاف بث الكاميرا.'})
        if interpreter:
            interpreter.reset_after_no_hand() # إعادة تعيين حالة المفسر عند الإيقاف
            if interpreter.sentence: # نطق أي جملة متبقية
                full_sentence = ' '.join(interpreter.sentence)
                print(f"نطق الجملة المتبقية عند الإيقاف: {full_sentence}")
                speak_elevenlabs(full_sentence)
                interpreter.sentence.clear()
    else:
        emit('camera_stream_stopped', {'status': 'info', 'message': 'بث الكاميرا ليس نشطاً.'})

@socketio.on('clear_sentence')
def handle_clear_sentence():
    global interpreter
    if interpreter:
        interpreter.clear_sentence()
    emit('sentence_cleared', {'status': 'success', 'sentence': ''}) # أرسل جملة فارغة لتحديث الواجهة

# تشغيل التطبيق
if __name__ == '__main__':
    # التأكد من وجود المجلدات المطلوبة
    for folder in ['templates', 'static']:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"تم إنشاء مجلد: {folder}")
    
    # التحقق من وجود ملفات النموذج والخط العربي
    required_files = ['test.h5', 'test_actions.txt', 'test_scaler_mean.npy', 'test_scaler_scale.npy']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            
    font_path_check = os.path.join(os.path.dirname(__file__), "NotoNaskhArabic-Regular.ttf")
    if not os.path.exists(font_path_check):
        print("⚠️ تحذير: ملف الخط العربي 'NotoNaskhArabic-Regular.ttf' غير موجود في نفس مجلد 'app.py'. قد لا يتم عرض النص العربي بشكل صحيح.")
        print("  يمكنك تحميله من: https://fonts.google.com/noto/specimen/Noto+Naskh+Arabic")
        print("  ثم ضع الملف NotoNaskhArabic-Regular.ttf في نفس مجلد app.py")

    if missing_files:
        print("⚠️  ملفات النموذج مفقودة:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nتأكد من وجود هذه الملفات في المجلد الرئيسي للمشروع.")
    else:
        print("✅ جميع ملفات النموذج المطلوبة موجودة.")
    
    print("\n🚀 بدء تشغيل الخادم...")
    print(f"🌐 سيكون الخادم متاحاً على: http://localhost:5001 (الكاميرا الافتراضية المستخدمة: {CAMERA_INDEX})")
    print("📱 أو على شبكتك المحلية: http://YOUR_IP:5001")
    
    try:
        # تشغيل SocketIO app
        socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n👋 تم إغلاق الخادم يدويًا.")
    except Exception as e:
        print(f"❌ خطأ فادح في تشغيل الخادم: {e}")
