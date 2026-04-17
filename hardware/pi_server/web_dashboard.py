import asyncio
import csv
import datetime
import glob
import io
import json
import os
import socket
import threading
import time
from collections import Counter, deque

import librosa
import numpy as np
import onnxruntime as ort
import scipy.io.wavfile as wavfile
import scipy.signal
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

HOST = "0.0.0.0"
PORT = 5000
SAMPLE_RATE = 32000
DURATION = 5
GAIN = 30.0

b_filter, a_filter = scipy.signal.butter(4, [1000, 12000], btype="bandpass", fs=SAMPLE_RATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

DB_FILE = os.path.join(BASE_DIR, "lifetime_stats.json")
CONFIG_FILE = os.path.join(BASE_DIR, "settings.json")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

current_config = {
    "threshold": 0.05,
    "alwayson": False,
    "smoothing": 0.5,
    "retentiondays": 7,
    "latitude": 49.4521,
    "longitude": 11.0767,
    "modelfile": "efficientnet_fp32.onnx",
    "classesfile": "efficientnet_task-species_ch-3_attn-False_layers-3_classes.json",
    "typemodelfile": "efficientnet_type_fp32.onnx", # Add this ONNX file later for the feature to activate
    "typeclassesfile": "efficientnet_task-type_ch-3_size-224_attn-False_layers-3_classes.json",
}

esp32_data = {}
lifetime_stats = Counter()

active_date = datetime.datetime.now().strftime("%Y-%m-%d")
active_day_data = []

# Species Model Globals
session_species = None
input_name_species = None
classes_species = []

# Type Model Globals (Song vs Call)
session_type = None
input_name_type = None
classes_type = []


def save_lifetime_stats():
    with open(DB_FILE, "w") as f:
        json.dump(dict(lifetime_stats), f)


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(current_config, f)


def load_day_history(date_str: str):
    path = os.path.join(HISTORY_DIR, f"{date_str}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_day_history(date_str: str, data):
    path = os.path.join(HISTORY_DIR, f"{date_str}.json")
    with open(path, "w") as f:
        json.dump(data, f)


def clean_old_history():
    retention = int(current_config.get("retentiondays", 7))
    cutoff = time.time() - retention * 86400

    for fpath in glob.glob(os.path.join(HISTORY_DIR, "*.json")):
        try:
            if os.stat(fpath).st_mtime < cutoff:
                os.remove(fpath)
        except Exception:
            pass

    for fpath in glob.glob(os.path.join(AUDIO_DIR, "*.wav")):
        try:
            if os.stat(fpath).st_mtime < cutoff:
                os.remove(fpath)
        except Exception:
            pass


def log_detection(ip, bird, conf, vocal_type="-", audio_filename=None):
    global active_date, active_day_data

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")

    if today != active_date:
        save_day_history(active_date, active_day_data)
        clean_old_history()
        active_date = today
        active_day_data = load_day_history(today)

    entry = {
        "time": now.strftime("%H:%M:%S"),
        "bird": bird,
        "vocal_type": vocal_type,
        "conf": conf,
        "ip": ip,
    }

    if audio_filename:
        entry["audio"] = f"/audio/{audio_filename}"

    active_day_data.append(entry)
    save_day_history(today, active_day_data)
    return entry


def load_ai_model():
    global session_species, input_name_species, classes_species
    global session_type, input_name_type, classes_type

    # 1. Load Species Model
    model_path_species = os.path.join(MODELS_DIR, current_config.get("modelfile", ""))
    classes_path_species = os.path.join(MODELS_DIR, current_config.get("classesfile", ""))

    try:
        print(f"[*] Loading AI Model (Species): {model_path_species}...")
        session_species = ort.InferenceSession(model_path_species, providers=["CPUExecutionProvider"])
        input_name_species = session_species.get_inputs()[0].name

        with open(classes_path_species, "r") as f:
            data = json.load(f)

        classes_species = data["classes"] if isinstance(data, dict) and "classes" in data else data
        print(f"[+] Species Model loaded. ({len(classes_species)} classes)")
    except Exception as e:
        print(f"[-] Error loading species model: {e}")
        session_species = None
        input_name_species = None
        classes_species = []

    # 2. Load Type Model (Song vs Call)
    model_path_type = os.path.join(MODELS_DIR, current_config.get("typemodelfile", ""))
    classes_path_type = os.path.join(MODELS_DIR, current_config.get("typeclassesfile", ""))
    
    try:
        if os.path.exists(model_path_type) and os.path.exists(classes_path_type):
            print(f"[*] Loading AI Model (Type): {model_path_type}...")
            session_type = ort.InferenceSession(model_path_type, providers=["CPUExecutionProvider"])
            input_name_type = session_type.get_inputs()[0].name

            with open(classes_path_type, "r") as f:
                data = json.load(f)

            classes_type = data["classes"] if isinstance(data, dict) and "classes" in data else data
            print(f"[+] Type Model loaded. ({len(classes_type)} classes)")
        else:
            print("[-] Type model ONNX/classes not found. Skipping song/call detection feature.")
            session_type = None
            input_name_type = None
            classes_type = []
    except Exception as e:
        print(f"[-] Error loading type model: {e}")
        session_type = None
        input_name_type = None
        classes_type = []


def preprocess_audio(audio_float):
    audio_float = audio_float - np.mean(audio_float)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_float,
        sr=SAMPLE_RATE,
        n_mels=128,
        fmax=16000,
        hop_length=512,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    min_val, max_val = log_mel_spec.min(), log_mel_spec.max()
    spec_norm = (log_mel_spec - min_val) / (max_val - min_val + 1e-6)
    spec_norm = np.flipud(spec_norm).copy()
    spec_tensor = torch.tensor(spec_norm, dtype=torch.float32)
    tensor = spec_tensor.unsqueeze(0).repeat(3, 1, 1)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return tensor.numpy()


def filtered_audio_from_int16(mono_audio: np.ndarray) -> np.ndarray:
    audio_float = mono_audio.astype(np.float32) / 32768.0
    audio_float *= GAIN
    audio_float = scipy.signal.filtfilt(b_filter, a_filter, audio_float)
    audio_float = np.clip(audio_float, -1.0, 1.0)
    return audio_float


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def handle_client(conn, addr):
    ip = addr[0]
    print(f"[+] ESP32 {ip} connected!")

    esp32_data[ip] = {
        "rms": 0.0,
        "status": "Listening...",
        "prediction": "-",
    }

    bytes_per_second = SAMPLE_RATE * 2
    buffer_max = bytes_per_second * DURATION
    ui_chunk = bytes_per_second
    process_every_bytes = bytes_per_second

    raw_buffer = bytearray()
    bytes_since_last_process = 0
    last_preds = deque(maxlen=3)
    last_preds_type = deque(maxlen=3)
    last_seen_state = {"bird": None, "time": 0.0}

    ignore_words = [
        "cricket", "frog", "toad", "cicada", "bat", "grasshopper",
        "insect", "noise", "locust", "katydid", "roesel", "shrew", "mouse",
    ]

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break

            raw_buffer.extend(data)
            bytes_since_last_process += len(data)

            if len(raw_buffer) > buffer_max:
                raw_buffer = raw_buffer[-buffer_max:]

            if len(raw_buffer) >= ui_chunk:
                mono_audio_ui = np.frombuffer(bytes(raw_buffer[-ui_chunk:]), dtype=np.int16)
                if len(mono_audio_ui) > 64:
                    try:
                        audio_float_ui = filtered_audio_from_int16(mono_audio_ui)
                        rms = float(np.sqrt(np.mean(audio_float_ui ** 2)))
                    except Exception:
                        rms = 0.0
                else:
                    rms = 0.0

                esp32_data[ip]["rms"] = rms
                if rms > current_config["threshold"] or current_config["alwayson"]:
                    esp32_data[ip]["status"] = "Recording... AI active..."
                else:
                    esp32_data[ip]["status"] = "Waiting for sound..."

            if len(raw_buffer) >= buffer_max and bytes_since_last_process >= process_every_bytes:
                bytes_since_last_process = 0

                if session_species is None or input_name_species is None or not classes_species:
                    continue

                mono_audio = np.frombuffer(bytes(raw_buffer), dtype=np.int16)
                if len(mono_audio) < 1024:
                    continue

                try:
                    audio_float = filtered_audio_from_int16(mono_audio)
                    window_rms = float(np.sqrt(np.mean(audio_float ** 2)))

                    if window_rms <= current_config["threshold"] and not current_config["alwayson"]:
                        continue

                    start_time = time.time()
                    input_tensor = preprocess_audio(audio_float)
                    
                    # 1. Inference: Species Model
                    outputs_species = session_species.run(None, {input_name_species: input_tensor})
                    logits_species = torch.tensor(outputs_species[0][0])
                    probs_species = F.softmax(logits_species, dim=0).numpy()

                    last_preds.append(probs_species)
                    avg_preds = np.mean(last_preds, axis=0)

                    smoothing = float(current_config.get("smoothing", 0.5))
                    smoothed_preds = avg_preds * smoothing + probs_species * (1.0 - smoothing)

                    best_idx = int(np.argmax(smoothed_preds))
                    confidence = float(smoothed_preds[best_idx])
                    bird_name = classes_species[best_idx]
                    is_ignored = any(word in bird_name.lower() for word in ignore_words)
                    
                    # 2. Inference: Type Model (Song vs Call)
                    vocal_type_str = "-"
                    if session_type is not None and input_name_type is not None:
                        outputs_type = session_type.run(None, {input_name_type: input_tensor})
                        logits_type = torch.tensor(outputs_type[0][0])
                        probs_type = F.softmax(logits_type, dim=0).numpy()
                        
                        last_preds_type.append(probs_type)
                        avg_preds_type = np.mean(last_preds_type, axis=0)
                        smoothed_preds_type = avg_preds_type * smoothing + probs_type * (1.0 - smoothing)
                        
                        best_type_idx = int(np.argmax(smoothed_preds_type))
                        vocal_type_str = classes_type[best_type_idx].capitalize()

                    duration_ms = int((time.time() - start_time) * 1000)

                    if confidence > 0.5 and not is_ignored:
                        current_time = time.time()
                        is_new_encounter = True

                        if (
                            last_seen_state["bird"] == bird_name
                            and current_time - last_seen_state["time"] < 10.0
                        ):
                            is_new_encounter = False

                        last_seen_state["bird"] = bird_name
                        last_seen_state["time"] = current_time

                        if vocal_type_str != "-":
                            pred_str = f"{bird_name} ({vocal_type_str}) - {confidence * 100:.1f}%"
                        else:
                            pred_str = f"{bird_name} ({confidence * 100:.1f}%)"
                            
                        esp32_data[ip]["prediction"] = pred_str

                        audio_filename = None
                        if confidence >= 0.85:
                            ts = time.strftime("%H-%M-%S")
                            clean_name = safe_name(bird_name)
                            audio_filename = f"{active_date}_{ts}_{clean_name}.wav"
                            audio_path = os.path.join(AUDIO_DIR, audio_filename)
                            audio_final_int16 = (audio_float * 32767.0).astype(np.int16)
                            wavfile.write(audio_path, SAMPLE_RATE, audio_final_int16)

                        log_detection(ip, bird_name, confidence * 100, vocal_type_str, audio_filename)
                        print(f"[{ip}] {pred_str} | {duration_ms} ms")

                        if is_new_encounter:
                            lifetime_stats[bird_name] += 1
                            save_lifetime_stats()

                except Exception as ai_error:
                    print(f"[{ip}] AI Error: {ai_error}")

    except Exception as e:
        print(f"[-] Connection to {ip} lost: {e}")
    finally:
        if ip in esp32_data:
            del esp32_data[ip]
        conn.close()


def start_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)

    print(f"[*] I2S Socket Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, "r") as f:
            lifetime_stats.update(json.load(f))
    except Exception:
        pass

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            current_config.update(json.load(f))
    except Exception:
        pass

active_day_data = load_day_history(active_date)
clean_old_history()
load_ai_model()

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Piepmatz Polizei Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #00A2E8;
            --bg: #ecf0f1;
            --card: #ffffff;
            --accent: #FF7F29;
        }

        body {
            font-family: Inter, sans-serif;
            background-color: var(--bg);
            margin: 0;
            padding: 0;
        }

        header {
            background-color: #ffffff;
            color: var(--primary);
            padding: 15px 30px;
            display: flex;
            align-items: center;
            gap: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-bottom: 3px solid var(--secondary);
        }

        .logo {
            height: 60px;
            object-fit: contain;
        }

        .tabs {
            display: flex;
            background: var(--primary);
            padding: 0 20px;
        }

        .tab-btn {
            background: none;
            border: none;
            color: #bdc3c7;
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
            font-family: Inter, sans-serif;
        }

        .tab-btn:hover {
            color: white;
        }

        .tab-btn.active {
            color: white;
            border-bottom: 3px solid var(--secondary);
            font-weight: 600;
        }

        .tab-content {
            display: none;
            padding: 30px;
            max-width: 1200px;
            margin: 0 auto;
        }

        .tab-content.active {
            display: block;
        }

        .card {
            background: var(--card);
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }

        .bar-container {
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 6px;
            height: 20px;
            margin-top: 10px;
        }

        .bar {
            height: 100%;
            background-color: var(--accent);
            width: 0;
            transition: width 0.1s ease;
            border-radius: 6px;
        }

        .bar.active {
            background-color: #e74c3c;
        }

        .prediction {
            font-size: 18px;
            color: var(--accent);
            font-weight: 700;
            margin-top: 10px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: var(--primary);
            color: white;
            font-weight: 600;
        }

        .chart-container {
            position: relative;
            height: 40vh;
            width: 100%;
            margin-top: 20px;
        }

        .config-row {
            margin-bottom: 20px;
        }

        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .date-picker, .btn {
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-family: Inter;
            font-size: 14px;
        }

        .btn {
            background: var(--primary);
            color: white;
            cursor: pointer;
            border: none;
            font-weight: 600;
        }

        .btn:hover {
            background: var(--secondary);
        }

        input[type=range] {
            accent-color: var(--accent);
        }

        audio {
            height: 30px;
        }

        #weather-stats {
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: #555;
            margin-top: 8px;
            flex-wrap: wrap;
        }
    </style>
</head>
<body>
    <header>
        <img src="/static/logo.png" alt="Logo" class="logo" onerror="this.style.display='none'">
        <h1>Piepmatz Polizei - AI Monitor</h1>
    </header>

    <div class="tabs">
        <button class="tab-btn active" onclick="openTab(event, 'History')">Day Overview</button>
        <button class="tab-btn" onclick="openTab(event, 'Stats')">Lifetime Stats</button>
        <button class="tab-btn" onclick="openTab(event, 'Live')">Live Radar</button>
        <button class="tab-btn" onclick="openTab(event, 'AI')">Settings</button>
    </div>

    <div id="History" class="tab-content active">
        <h2>Day Overview Map</h2>
        <div class="card">
            <div class="controls">
                <input type="date" id="history-date" class="date-picker" onchange="fetchDayHistory()">
                <label><input type="checkbox" id="toggle-activity" checked onchange="toggleChartElements()"> Show Overall Activity</label>
                <label><input type="checkbox" id="toggle-temp" onchange="toggleWeather()"> Show Temperature</label>
                <label><input type="checkbox" id="toggle-rain" onchange="toggleWeather()"> Show Rain</label>

                <select id="export-range" class="date-picker" style="margin-left:auto;">
                    <option value="today">Export Selected Day</option>
                    <option value="7days">Export Last 7 Days</option>
                    <option value="all">Export All History</option>
                </select>
                <button class="btn" onclick="exportCSV()">Download CSV</button>
            </div>

            <div class="chart-container" style="height: 50vh;">
                <canvas id="timelineChart"></canvas>
            </div>
            <div id="weather-stats"></div>

            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Species</th>
                        <th>Type</th>
                        <th>Confidence</th>
                        <th>Audio Replay</th>
                    </tr>
                </thead>
                <tbody id="history-table">
                    <tr><td colspan="5">No data available</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div id="Stats" class="tab-content">
        <h2>Total Detections Lifetime</h2>
        <div class="card">
            <div class="chart-container">
                <canvas id="speciesChart"></canvas>
            </div>
        </div>
    </div>

    <div id="Live" class="tab-content">
        <h2>Active Audio Streams</h2>
        <div id="nodes"><i>Waiting for ESP32 connections...</i></div>
    </div>

    <div id="AI" class="tab-content">
        <div class="card">
            <h3>AI Settings</h3>

            <div class="config-row">
                <label><b>Always-On Mode</b></label><br>
                <input type="checkbox" id="alwayson" onchange="sendConfig()"> Enable continuous inference
            </div>

            <div class="config-row">
                <label><b>RMS Threshold (Wake-on-Sound)</b> <span id="thresh-val">0.05</span></label><br>
                <input
                    type="range"
                    id="threshold"
                    min="0.001"
                    max="0.3"
                    step="0.005"
                    value="0.05"
                    style="width: 300px;"
                    onchange="sendConfig()"
                    oninput="document.getElementById('thresh-val').innerText=this.value">
            </div>

            <div class="config-row">
                <label><b>Temporal Smoothing Factor</b> <span id="smooth-val">0.5</span></label><br>
                <input
                    type="range"
                    id="smoothing"
                    min="0"
                    max="1"
                    step="0.1"
                    value="0.5"
                    style="width: 300px;"
                    onchange="sendConfig()"
                    oninput="document.getElementById('smooth-val').innerText=this.value">
            </div>

            <hr>

            <h3>Storage / Location</h3>

            <div class="config-row">
                <label><b>History Retention Days</b> <span id="retention-val">7</span></label><br>
                <input
                    type="number"
                    id="retention"
                    min="1"
                    max="365"
                    value="7"
                    style="width: 100px; padding: 5px;"
                    onchange="document.getElementById('retention-val').innerText=this.value; sendConfig()">
            </div>

            <div class="config-row">
                <label><b>Latitude for Weather</b></label>
                <input type="number" id="lat" value="49.4521" step="0.0001" style="width: 120px; padding: 5px;" onchange="sendConfig()">

                <label style="margin-left: 10px;"><b>Longitude</b></label>
                <input type="number" id="lon" value="11.0767" step="0.0001" style="width: 120px; padding: 5px;" onchange="sendConfig()">
            </div>
        </div>
    </div>

    <script>
        let configLoaded = false;
        let currentHistoryLength = -1;
        let currentRenderedDate = null;
        let weatherDataCache = {};
        let chartInstance = null;
        let timelineChart = null;

        const colorPalette = [
            "#00A2E8", "#FF7F29", "#E74C3C", "#2ECC71",
            "#9B59B6", "#F1C40F", "#34495E", "#E67E22", "#16A085"
        ];

        let birdColorMap = {};

        function getLocalDateString() {
            const now = new Date();
            const year = now.getFullYear();
            const month = String(now.getMonth() + 1).padStart(2, '0');
            const day = String(now.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
        }

        function getBirdColor(birdName) {
            if (!birdColorMap[birdName]) {
                birdColorMap[birdName] = colorPalette[Object.keys(birdColorMap).length % colorPalette.length];
            }
            return birdColorMap[birdName];
        }

        document.getElementById("history-date").value = getLocalDateString();

        function openTab(evt, tabName) {
            const tabcontent = document.getElementsByClassName("tab-content");
            const tablinks = document.getElementsByClassName("tab-btn");

            for (let i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
                tabcontent[i].classList.remove("active");
            }

            for (let i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }

            document.getElementById(tabName).style.display = "block";
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.className += " active";
        }

        const ws = new WebSocket(`ws://${location.host}/ws`);

        function sendConfig() {
            ws.send(JSON.stringify({
                type: "config",
                alwayson: document.getElementById("alwayson").checked,
                threshold: parseFloat(document.getElementById("threshold").value),
                smoothing: parseFloat(document.getElementById("smoothing").value),
                retentiondays: parseInt(document.getElementById("retention").value),
                latitude: parseFloat(document.getElementById("lat").value),
                longitude: parseFloat(document.getElementById("lon").value)
            }));
        }

        function exportCSV() {
            let range = document.getElementById("export-range").value;
            if (range === "today") {
                range = document.getElementById("history-date").value;
            }
            window.location.href = `/api/export?range=${encodeURIComponent(range)}`;
        }

        async function fetchDayHistory() {
            const dateStr = document.getElementById("history-date").value;
            const res = await fetch(`/api/history/${dateStr}`);
            const data = await res.json();

            if (document.getElementById("toggle-temp").checked || document.getElementById("toggle-rain").checked) {
                await fetchWeatherData(dateStr);
            }

            renderDayData(data, dateStr, true);
        }

        async function fetchWeatherData(dateStr) {
            if (weatherDataCache[dateStr]) return;

            const lat = document.getElementById("lat").value;
            const lon = document.getElementById("lon").value;
            const todayStr = getLocalDateString();

            try {
                const isTodayOrFuture = dateStr >= todayStr;
                const baseUrl = isTodayOrFuture
                    ? "https://api.open-meteo.com/v1/forecast"
                    : "https://archive-api.open-meteo.com/v1/archive";

                const url = `${baseUrl}?latitude=${lat}&longitude=${lon}&hourly=temperature_2m,precipitation&start_date=${dateStr}&end_date=${dateStr}`;
                const res = await fetch(url);
                const json = await res.json();

                if (!json.error && json.hourly) {
                    weatherDataCache[dateStr] = {
                        temp: json.hourly.temperature_2m || [],
                        rain: json.hourly.precipitation || []
                    };
                    return;
                }
            } catch (e) {
                console.warn("Primary weather fetch failed", e);
            }

            try {
                const fallbackUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&hourly=temperature_2m,precipitation&start_date=${dateStr}&end_date=${dateStr}`;
                const res = await fetch(fallbackUrl);
                const json = await res.json();

                if (!json.error && json.hourly) {
                    weatherDataCache[dateStr] = {
                        temp: json.hourly.temperature_2m || [],
                        rain: json.hourly.precipitation || []
                    };
                }
            } catch (e) {
                console.warn("Fallback weather fetch failed", e);
            }
        }

        async function toggleWeather() {
            const dateStr = document.getElementById("history-date").value;
            if (document.getElementById("toggle-temp").checked || document.getElementById("toggle-rain").checked) {
                await fetchWeatherData(dateStr);
            }
            await fetchDayHistory();
        }

        function toggleChartElements() {
            if (!timelineChart) return;

            const showAct = document.getElementById("toggle-activity").checked;
            const showTemp = document.getElementById("toggle-temp").checked;
            const showRain = document.getElementById("toggle-rain").checked;

            timelineChart.data.datasets.forEach(ds => {
                if (ds.label === "Overall Activity") ds.hidden = !showAct;
                if (ds.label === "Temperature (°C)") ds.hidden = !showTemp;
                if (ds.label === "Precipitation (mm)") ds.hidden = !showRain;
            });

            if (timelineChart.options.scales.yActivity) {
                timelineChart.options.scales.yActivity.display = showAct;
            }
            if (timelineChart.options.scales.yTemp) {
                timelineChart.options.scales.yTemp.display = showTemp;
            }
            if (timelineChart.options.scales.yRain) {
                timelineChart.options.scales.yRain.display = showRain;
            }

            timelineChart.update("none");
        }

        function updateStatsChart(statsData) {
            const ctx = document.getElementById("speciesChart").getContext("2d");

            const labels = Object.keys(statsData);
            const data = Object.values(statsData);
            const bgColors = labels.map(b => getBirdColor(b));

            if (chartInstance) {
                chartInstance.data.labels = labels;
                chartInstance.data.datasets[0].data = data;
                chartInstance.data.datasets[0].backgroundColor = bgColors;
                chartInstance.update("none");
            } else {
                chartInstance = new Chart(ctx, {
                    type: "bar",
                    data: {
                        labels: labels,
                        datasets: [{
                            data: data,
                            backgroundColor: bgColors
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: false,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: { stepSize: 1 }
                            }
                        }
                    }
                });
            }
        }

        function renderDayData(historyData, dateStr, forceRedraw=false) {
            if (!forceRedraw && historyData.length === currentHistoryLength && dateStr === currentRenderedDate) {
                return;
            }

            currentHistoryLength = historyData.length;
            currentRenderedDate = dateStr;

            const ctxT = document.getElementById("timelineChart").getContext("2d");
            const statsDiv = document.getElementById("weather-stats");
            const datasets = [];
            const labelsY = [];
            const groupedByBird = {};
            const hourlyCounts = new Array(24).fill(0);

            let tempMin = null;
            let tempMax = null;
            let rainMax = null;

            if (statsDiv) statsDiv.innerHTML = "";

            historyData.forEach(item => {
                if (!labelsY.includes(item.bird)) labelsY.push(item.bird);
                if (!groupedByBird[item.bird]) groupedByBird[item.bird] = [];

                const fullIsoString = `${dateStr}T${item.time}`;
                groupedByBird[item.bird].push({
                    x: fullIsoString,
                    y: item.bird,
                    conf: item.conf
                });

                const hour = parseInt(item.time.split(":")[0], 10);
                if (!Number.isNaN(hour) && hour >= 0 && hour < 24) {
                    hourlyCounts[hour] += 1;
                }
            });

            const activityData = [];
            for (let i = 0; i < 24; i++) {
                const h = i.toString().padStart(2, "0");
                activityData.push({ x: `${dateStr}T${h}:30:00`, y: hourlyCounts[i] });
            }

            datasets.push({
                type: "line",
                label: "Overall Activity",
                data: activityData,
                yAxisID: "yActivity",
                borderColor: "rgba(0,0,0,0.15)",
                backgroundColor: "rgba(0,0,0,0.05)",
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                hidden: !document.getElementById("toggle-activity").checked
            });

            if (weatherDataCache[dateStr]) {
                const weather = weatherDataCache[dateStr];

                if (weather.temp && weather.temp.length) {
                    const tempData = [];
                    for (let i = 0; i < Math.min(24, weather.temp.length); i++) {
                        const h = i.toString().padStart(2, "0");
                        tempData.push({ x: `${dateStr}T${h}:30:00`, y: weather.temp[i] });
                    }

                    datasets.push({
                        type: "line",
                        label: "Temperature (°C)",
                        data: tempData,
                        yAxisID: "yTemp",
                        borderColor: "#e74c3c",
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.35,
                        pointRadius: 0,
                        hidden: !document.getElementById("toggle-temp").checked
                    });

                    const temps = weather.temp.filter(v => v !== null && v !== undefined);
                    if (temps.length) {
                        const tMin = Math.min(...temps);
                        const tMax = Math.max(...temps);
                        tempMin = Math.floor(tMin - 1);
                        tempMax = Math.ceil(tMax + 1);

                        if (statsDiv) {
                            statsDiv.innerHTML += `<span>Temp min/max: <b>${tMin.toFixed(1)}°C</b> / <b>${tMax.toFixed(1)}°C</b></span>`;
                        }
                    }
                }

                if (weather.rain && weather.rain.length) {
                    const rainData = [];
                    for (let i = 0; i < Math.min(24, weather.rain.length); i++) {
                        const h = i.toString().padStart(2, "0");
                        rainData.push({ x: `${dateStr}T${h}:30:00`, y: weather.rain[i] });
                    }

                    datasets.push({
                        type: "bar",
                        label: "Precipitation (mm)",
                        data: rainData,
                        yAxisID: "yRain",
                        backgroundColor: "rgba(0, 162, 232, 0.35)",
                        borderColor: "#00A2E8",
                        borderWidth: 1,
                        hidden: !document.getElementById("toggle-rain").checked
                    });

                    const rains = weather.rain.filter(v => v !== null && v !== undefined);
                    if (rains.length) {
                        const rMin = Math.min(...rains);
                        const rMax = Math.max(...rains);
                        rainMax = Math.max(1, Math.ceil(rMax + 1));

                        if (statsDiv) {
                            statsDiv.innerHTML += `<span>Rain min/max: <b>${rMin.toFixed(1)} mm</b> / <b>${rMax.toFixed(1)} mm</b></span>`;
                        }
                    }
                }
            }

            Object.keys(groupedByBird).forEach((birdName) => {
                datasets.push({
                    type: "scatter",
                    label: birdName,
                    data: groupedByBird[birdName],
                    yAxisID: "y",
                    backgroundColor: getBirdColor(birdName),
                    borderColor: getBirdColor(birdName),
                    pointRadius: 5,
                    pointHoverRadius: 6
                });
            });

            if (timelineChart) {
                timelineChart.destroy();
            }

            timelineChart = new Chart(ctxT, {
                data: {
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: {
                            type: "time",
                            time: {
                                unit: "hour",
                                displayFormats: { hour: "HH:mm" }
                            },
                            min: `${dateStr}T00:00:00`,
                            max: `${dateStr}T23:59:59`,
                            title: {
                                display: true,
                                text: "Time of Day",
                                font: { family: "Inter", weight: "bold" }
                            }
                        },
                        y: {
                            type: "category",
                            labels: labelsY,
                            position: "left",
                            grid: {
                                color: "rgba(0,0,0,0.05)"
                            },
                            title: {
                                display: true,
                                text: "Species",
                                font: { family: "Inter", weight: "bold" }
                            }
                        },
                        yActivity: {
                            type: "linear",
                            position: "right",
                            display: document.getElementById("toggle-activity").checked,
                            min: 0,
                            grid: {
                                drawOnChartArea: false
                            },
                            title: {
                                display: document.getElementById("toggle-activity").checked,
                                text: "Calls / hour"
                            }
                        },
                        yTemp: {
                            type: "linear",
                            position: "right",
                            display: document.getElementById("toggle-temp").checked,
                            min: tempMin !== null ? tempMin : undefined,
                            max: tempMax !== null ? tempMax : undefined,
                            grid: {
                                drawOnChartArea: false
                            },
                            title: {
                                display: document.getElementById("toggle-temp").checked,
                                text: "Temperature (°C)"
                            }
                        },
                        yRain: {
                            type: "linear",
                            position: "right",
                            offset: true,
                            display: document.getElementById("toggle-rain").checked,
                            min: 0,
                            max: rainMax !== null ? rainMax : undefined,
                            grid: {
                                drawOnChartArea: false
                            },
                            title: {
                                display: document.getElementById("toggle-rain").checked,
                                text: "Rain (mm)"
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    if (context.dataset.label === "Overall Activity") {
                                        return `${context.raw.y} calls this hour`;
                                    }
                                    if (context.dataset.label === "Temperature (°C)") {
                                        return `${context.raw.y} °C`;
                                    }
                                    if (context.dataset.label === "Precipitation (mm)") {
                                        return `${context.raw.y} mm`;
                                    }
                                    const t = new Date(context.raw.x).toLocaleTimeString();
                                    return `${context.dataset.label} (${context.raw.conf?.toFixed?.(1) ?? "?"}%) at ${t}`;
                                }
                            }
                        }
                    }
                }
            });

            const histTbody = document.getElementById("history-table");
            histTbody.innerHTML = "";

            if (historyData.length === 0) {
                histTbody.innerHTML = "<tr><td colspan='5'>No data available</td></tr>";
            } else {
                const reverseData = [...historyData].reverse();
                reverseData.forEach(item => {
                    const tr = document.createElement("tr");
                    const audioHtml = item.audio
                        ? `<audio controls src="${item.audio}"></audio>`
                        : `<span style="color:#aaa;">---</span>`;
                    
                    const typeStr = item.vocal_type ? item.vocal_type : "-";

                    tr.innerHTML = `
                        <td>${item.time}</td>
                        <td>${item.bird}</td>
                        <td>${typeStr}</td>
                        <td>${Number(item.conf).toFixed(1)}</td>
                        <td>${audioHtml}</td>
                    `;
                    histTbody.appendChild(tr);
                });
            }
        }

        ws.onmessage = async function(event) {
            const payload = JSON.parse(event.data);

            if (!configLoaded && payload.config) {
                document.getElementById("alwayson").checked = payload.config.alwayson;
                document.getElementById("threshold").value = payload.config.threshold;
                document.getElementById("thresh-val").innerText = payload.config.threshold;
                document.getElementById("smoothing").value = payload.config.smoothing;
                document.getElementById("smooth-val").innerText = payload.config.smoothing;
                document.getElementById("retention").value = payload.config.retentiondays || 7;
                document.getElementById("retention-val").innerText = payload.config.retentiondays || 7;
                if (payload.config.latitude !== undefined) document.getElementById("lat").value = payload.config.latitude;
                if (payload.config.longitude !== undefined) document.getElementById("lon").value = payload.config.longitude;
                configLoaded = true;
                await fetchDayHistory();
            }

            const nodesDiv = document.getElementById("nodes");
            if (Object.keys(payload.nodes).length === 0) {
                nodesDiv.innerHTML = "<i>Waiting for ESP32 connections...</i>";
            } else {
                nodesDiv.innerHTML = "";
                for (const ip in payload.nodes) {
                    const node = payload.nodes[ip];
                    const percent = Math.min(100, node.rms / 0.2 * 100);
                    const barClass = (node.rms > payload.config.threshold || payload.config.alwayson) ? "bar active" : "bar";

                    const card = document.createElement("div");
                    card.className = "card";
                    card.innerHTML = `
                        <h3>Node IP: ${ip}</h3>
                        <p><b>Status:</b> ${node.status} | <b>RMS:</b> ${Number(node.rms).toFixed(3)}</p>
                        <div class="bar-container"><div class="${barClass}" style="width:${percent}%"></div></div>
                        <div class="prediction">Last Detection: ${node.prediction}</div>
                    `;
                    nodesDiv.appendChild(card);
                }
            }

            const selectedDate = document.getElementById("history-date").value;
            if (selectedDate === getLocalDateString() && payload.todaydata) {
                if (document.getElementById("toggle-temp").checked || document.getElementById("toggle-rain").checked) {
                    if (!weatherDataCache[selectedDate]) {
                        await fetchWeatherData(selectedDate);
                    }
                }
                renderDayData(payload.todaydata, selectedDate, false);
            }

            if (Object.keys(payload.stats).length > 0) {
                updateStatsChart(payload.stats);
            }
        };
    </script>
</body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html_content)


@app.get("/api/history/{date_str}")
async def get_history(date_str: str):
    return load_day_history(date_str)


@app.get("/api/export")
async def export_data(range: str = "today"):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Time", "Bird Species", "Vocalization Type", "Confidence", "Audio Link"])

    files_to_read = []

    if range == "all":
        files_to_read = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    elif range == "7days":
        now = time.time()
        for fpath in glob.glob(os.path.join(HISTORY_DIR, "*.json")):
            try:
                if os.stat(fpath).st_mtime >= now - 7 * 86400:
                    files_to_read.append(fpath)
            except Exception:
                pass
    else:
        target_file = os.path.join(HISTORY_DIR, f"{range}.json")
        if os.path.exists(target_file):
            files_to_read.append(target_file)

    for filepath in sorted(files_to_read):
        date_str = os.path.basename(filepath).replace(".json", "")
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            for item in data:
                writer.writerow([
                    date_str,
                    item.get("time", ""),
                    item.get("bird", ""),
                    item.get("vocal_type", "-"),
                    item.get("conf", ""),
                    item.get("audio", ""),
                ])
        except Exception:
            pass

    filename = f"birddata_{range}.csv"
    return PlainTextResponse(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                if data.get("type") == "config":
                    current_config["alwayson"] = data.get("alwayson", False)
                    current_config["threshold"] = data.get("threshold", 0.05)
                    current_config["smoothing"] = data.get("smoothing", 0.5)
                    current_config["retentiondays"] = data.get("retentiondays", 7)

                    if "latitude" in data:
                        current_config["latitude"] = data["latitude"]
                    if "longitude" in data:
                        current_config["longitude"] = data["longitude"]

                    save_config()
                    clean_old_history()
            except asyncio.TimeoutError:
                pass

            payload = {
                "nodes": esp32_data,
                "todaydata": active_day_data,
                "config": current_config,
                "stats": dict(lifetime_stats),
            }

            try:
                await websocket.send_json(payload)
            except Exception:
                break
    except Exception:
        pass


if __name__ == "__main__":
    threading.Thread(target=start_socket_server, daemon=True).start()
    print("[*] Starting Web-Dashboard on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
