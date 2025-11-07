import os
import json
import joblib
import numpy as np
import logging
import tensorflow as tf
from flask import Flask, request, send_file
import openai
import tempfile
from gtts import gTTS

# =========================== CONFIG ===========================
MODEL_PATH = "./cloud_model.keras"
SCALER_PATH = "./scaler.pkl"
LABELS_PATH = "./label_encoder.json"
FEATURES_PATH = "./features.json"
PORT = int(os.environ.get("PORT", 5000))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("CLOUD_API")

# ===================== LOAD ARTIFACTS =====================
scaler = joblib.load(SCALER_PATH)
with open(LABELS_PATH) as f:
    CLASSES = json.load(f)
with open(FEATURES_PATH) as f:
    FEATURES = json.load(f)
model = tf.keras.models.load_model(MODEL_PATH)
FEATURES_COUNT = len(FEATURES)
CLASSES_COUNT = len(CLASSES)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

def jarvis_gpt_analysis(probabilities, sensor_values):
    if not OPENAI_API_KEY:
        return "GPT analysis unavailable (OpenAI API key not set)."
    fault_probs_dict = {name: float(prob) for name, prob in zip(CLASSES, probabilities)}
    feature_dict = {name: float(val) for name, val in zip(FEATURES, sensor_values)}
    prompt = (
        f"You are Jarvis, an expert industrial AI. "
        f"Based on fault probabilities {fault_probs_dict} and sensor readings {feature_dict}, "
        "give a concise, professional, qualitative health assessment: "
        "\"Equipment is at moderate/high/low risk; monitor sensors X,Y; predicted time until maintenance...\""
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Jarvis, expert industrial maintenance AI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT analysis failed: {e}")
        return "GPT analysis unavailable."

@app.route('/', methods=['GET'])
def home():
    return (
        "Cloud Model Online.\n"
        f"POST a single CSV row ({FEATURES_COUNT} comma-separated values, NOT JSON) to /predict for fault diagnosis.\n"
        "Handles raw sensor readings, scales, predicts, and returns text/voice Jarvis analysis.\n"
        "POST plain text to /tts for audio streaming (GPT analysis spoken as MP3).\n"
        "GET /tts?text=Your%20Text%20Here for streaming MP3 audio of spoken text (for ESP32 audio modules).\n"
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw = request.data.decode().strip()
        vals = raw.split(',')
        if len(vals) != FEATURES_COUNT:
            return (
                f"ERROR: Expected {FEATURES_COUNT} features (got {len(vals)})."
                " Please send a full CSV row with correct feature count.\n"
            ), 400
        sensor_data = np.array([float(x) for x in vals]).reshape(1, -1)
        sensor_data_scaled = scaler.transform(sensor_data)
        fault_probs = model.predict(sensor_data_scaled)[0]
        results = f"INPUT FEATURES ({FEATURES_COUNT})\n-------------------\n"
        for f, v in zip(FEATURES, sensor_data[0]):
            results += f"  {f}: {v:.6f}\n"
        results += "-------------------\nAI RESULT\n---------\n"
        max_idx = int(np.argmax(fault_probs))
        max_prob = fault_probs[max_idx]
        results += f"Fault: {CLASSES[max_idx]}\nConfidence: {max_prob*100:.2f}%\n\n"
        results += "Confidence Breakdown:\n"
        for i, p in enumerate(fault_probs):
            results += f"  {CLASSES[i]}: {p*100:.2f}%\n"
        results += "---------------------\n"
        jarvis_text = jarvis_gpt_analysis(fault_probs, sensor_data_scaled[0])
        results += "\nJarvis GPT Analysis:\n" + jarvis_text + "\n"
        return results, 200
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return str(e), 500

@app.route('/tts', methods=['POST', 'GET'])
def text_to_speech():
    """
    POST: plain text body (Jarvis GPT analysis) --> returns MP3 audio.
    GET: /tts?text=Your%20Text%20Here --> returns MP3 audio (for ESP32 streaming).
    """
    try:
        if request.method == 'POST':
            analysis_text = request.data.decode().strip()
        else:  # GET
            analysis_text = request.args.get('text', '').strip()
        if not analysis_text:
            return "ERROR: No text provided for TTS.", 400
        # Google Text-to-Speech
        tts = gTTS(text=analysis_text, lang='en')
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_mp3.name)
        temp_mp3.close()
        return send_file(temp_mp3.name, mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        return str(e), 500

if __name__ == "__main__":
    logger.info("🚀 Cloud API starting (raw CSV row only, matches ESP32 prediction output layout).")
    app.run(host='0.0.0.0', port=PORT, debug=False)
