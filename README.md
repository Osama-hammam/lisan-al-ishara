# Lisan Al Ishara - Arabic Sign Language Translator

## Introduction

This project is an Arabic Sign Language translation system (Lisan Al Ishara) that uses machine learning and computer vision techniques to convert hand signs into text and speech, and vice versa.  
The system includes an interactive web interface built with Flask, leveraging libraries like MediaPipe and TensorFlow for sign recognition, alongside a 3D avatar model for animation.

---

## Project Structure

- `app.py`  
  The main Flask server application that connects the frontend with the backend.  
  It handles camera input, runs the trained model, and sends prediction results to the interface.

- `requirements.txt`  
  Lists the required Python packages for the project.

- `templates/index.html`  
  The web frontend containing tabs for "Speech to Sign" and "Sign to Speech," with avatar animations.

- `static/`  
  Contains static assets such as images (`.jpeg`) and 3D avatar model files (`.glb`).

- `model_files/`  
  Contains trained model files and normalization parameters.

- Trained model files:  
  - `test.h5`: The deep learning model for sign recognition.  
  - `test_actions.txt`: The list of recognized sign actions.  
  - `test_scaler_mean.npy` and `test_scaler_scale.npy`: Normalization parameters for input data.

- `NotoNaskhArabic-Regular.ttf`  
  Arabic font used for display purposes.

---

## How to Run

1. **Create a virtual environment and install dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. **Start the Flask application:**
python3 app.py

3. **Open your browser and navigate to:**
http://127.0.0.1:5000

4. **Use the application:**
- Speech to Sign Tab: Type text or use speech recognition to display sign animations via the avatar.
- Sign to Speech Tab: Use the webcam to capture hand signs; the system will recognize them and convert to text and speech.

---

## System Requirements

- Python 3.8 or newer
- Python packages listed in requirements.txt
- Modern web browser supporting WebGL and WebRTC (e.g., Chrome, Firefox)
- Webcam (for Sign to Speech tab)

---

## Notes

- Make sure all .glb avatar files are inside the static/ folder as referenced in the code.
- Place the trained model files (test.h5, normalization files) where app.py can access them.
- Grant camera permissions in your browser to use the sign recognition tab.

---

## Author

- Name: Osama Hammam
- GitHub: Osama-hammam
- Email: osamaali2742001@gmail.com
