import socket
import threading
import numpy as np
import librosa
import onnxruntime as ort
import asyncio
import os
import json
import time
import datetime
import glob
import io
import csv
import torch
import torch.nn.functional as F
import scipy.signal
import scipy.io.wavfile as wavfile
from collections import deque, Counter
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 5000
SAMPLE_RATE = 32000
DURATION = 5
GAIN = 30.0  

# Bandpass Filter (1kHz - 12kHz)
b_filter, a_filter = scipy.signal.butter(4, [1000, 12000], btype='bandpass', fs=SAMPLE_RATE)

# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))   # pi_server/
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))   # bird-project/
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODELS_DIR  = os.path.join(DATA_DIR, "models")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
AUDIO_DIR   = os.path.join(DATA_DIR, "audio")
DB_FILE     = os.path.join(DATA_DIR, "lifetimestats.json")
CONFIG_FILE = os.path.join(DATA_DIR, "settings.json")

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

current_config = {
    "threshold": 0.05,
    "always_on": False,
    "smoothing": 0.5,
    "retention_days": 7,
    "latitude": 49.4521,    # Nuremberg
    "longitude": 11.0767,   # Nuremberg
    "model_file": "efficientnet_fp32.onnx",
    "classes_file": "efficientnet_task-species_ch-3_attn-False_layers-3_classes.json"
}

# --- STATE & DATABASE ---
esp32_data = {}
lifetime_stats = Counter()
active_day_data = []
active_date = datetime.datetime.now().strftime("%Y-%m-%d")

DB_FILE = os.path.join(DATA_DIR, "lifetime_stats.json")
CONFIG_FILE = os.path.join(DATA_DIR, "settings.json")

# Load existing stats and configs
if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, "r") as f:
            lifetime_stats.update(json.load(f))
    except Exception: pass

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            current_config.update(json.load(f))
    except Exception: pass

def save_lifetime_stats():
    with open(DB_FILE, "w") as f:
        json.dump(dict(lifetime_stats), f)

def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(current_config, f)

def load_day_history(date_str):
    path = os.path.join(HISTORY_DIR, f"{date_str}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_day_history(date_str, data):
    path = os.path.join(HISTORY_DIR, f"{date_str}.json")
    with open(path, "w") as f:
        json.dump(data, f)

def clean_old_history():
    retention = current_config.get("retention_days", 7)
    now = time.time()
    cutoff = now - (retention * 86400)
    
    # Delete old JSON logs
    for f in glob.glob(os.path.join(HISTORY_DIR, "*.json")):
        if os.stat(f).st_mtime < cutoff:
            try: os.remove(f)
            except: pass
            
    # Delete old Audio snippets
    for f in glob.glob(os.path.join(AUDIO_DIR, "*.wav")):
        if os.stat(f).st_mtime < cutoff:
            try: os.remove(f)
            except: pass

active_day_data = load_day_history(active_date)
clean_old_history()

def log_detection(ip, bird, conf, audio_filename=None):
    global active_date, active_day_data
    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    
    if today != active_date:
        save_day_history(active_date, active_day_data)
        clean_old_history()
        active_date = today
        active_day_data = load_day_history(today)
        
    entry = {"time": now.strftime("%H:%M:%S"), "bird": bird, "conf": conf, "ip": ip}
    if audio_filename:
        entry["audio"] = f"/audio/{audio_filename}"
        
    active_day_data.append(entry)
    save_day_history(today, active_day_data)
    return entry

# --- AI MODEL LOAD ---
session = None
input_name = None
classes = []

def load_ai_model():
    global session, input_name, classes
    model_path = os.path.join(MODELS_DIR, current_config["model_file"])
    classes_path = os.path.join(MODELS_DIR, current_config["classes_file"])
    try:
        print(f"[*] Loading AI Model: {model_path}...")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name

        with open(classes_path, "r") as f:
            data = json.load(f)
            classes = data["classes"] if isinstance(data, dict) and "classes" in data else data

        print(f"[+] Model loaded. ({len(classes)} classes)")
    except Exception as e:
        print(f"[-] Error loading model: {e}")

load_ai_model()

def preprocess_audio(audio_float):
    audio_float = audio_float - np.mean(audio_float)
    melspec = librosa.feature.melspectrogram(
        y=audio_float, sr=SAMPLE_RATE, n_mels=128, fmax=16000, hop_length=512
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    min_val, max_val = log_melspec.min(), log_melspec.max()
    spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)

    spec_norm = np.flipud(spec_norm).copy()
    spec_tensor = torch.tensor(spec_norm, dtype=torch.float32)
    tensor = spec_tensor.unsqueeze(0).repeat(3, 1, 1)
    
    tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    return tensor.numpy()

# --- I2S SOCKET SERVER ---
def handle_client(conn, addr):
    ip = addr[0]
    print(f"\n[+] ESP32 {ip} connected!")
    esp32_data[ip] = {"rms": 0.0, "status": "Listening...", "prediction": "-"}
    
    BYTES_PER_SECOND = SAMPLE_RATE * 2 
    BUFFER_MAX = BYTES_PER_SECOND * DURATION 
    raw_buffer = bytearray()
    last_preds = deque(maxlen=3)
    
    bytes_since_last_process = 0
    PROCESS_EVERY_BYTES = BYTES_PER_SECOND 
    last_seen_state = {"bird": None, "time": 0.0}

    try:
        while True:
            data = conn.recv(4096)
            if not data: break
            
            raw_buffer.extend(data)
            bytes_since_last_process += len(data)

            if len(raw_buffer) > BUFFER_MAX:
                raw_buffer = raw_buffer[-BUFFER_MAX:]

            ui_chunk = BYTES_PER_SECOND
            if len(raw_buffer) >= ui_chunk:
                mono_audio = np.frombuffer(bytes(raw_buffer[-ui_chunk:]), dtype=np.int16)
                
                audio_float_ui = (mono_audio.astype(np.float32) / 32768.0) * GAIN
                audio_float_ui = scipy.signal.filtfilt(b_filter, a_filter, audio_float_ui)
                audio_float_ui = np.clip(audio_float_ui, -1.0, 1.0)
                
                rms = float(np.sqrt(np.mean(audio_float_ui**2)))
                esp32_data[ip]["rms"] = rms

                if rms > current_config["threshold"] or current_config["always_on"]:
                    esp32_data[ip]["status"] = "Recording & AI active..."
                else:
                    esp32_data[ip]["status"] = "Waiting for sound..."

            if len(raw_buffer) == BUFFER_MAX and bytes_since_last_process >= PROCESS_EVERY_BYTES:
                bytes_since_last_process = 0 
                
                mono_audio = np.frombuffer(bytes(raw_buffer), dtype=np.int16)
                
                audio_float = (mono_audio.astype(np.float32) / 32768.0) * GAIN
                audio_float = scipy.signal.filtfilt(b_filter, a_filter, audio_float)
                audio_float = np.clip(audio_float, -1.0, 1.0)
                
                window_rms = float(np.sqrt(np.mean(audio_float**2)))

                if window_rms > current_config["threshold"] or current_config["always_on"]:
                    try:
                        input_tensor = preprocess_audio(audio_float)
                        start_time = time.time()
                        outputs = session.run(None, {input_name: input_tensor})

                        logits = torch.tensor(outputs[0][0])
                        probs = F.softmax(logits, dim=0).numpy()

                        last_preds.append(probs)
                        smoothed_preds = np.mean(last_preds, axis=0) * current_config["smoothing"] + probs * (1 - current_config["smoothing"])

                        best_idx = int(np.argmax(smoothed_preds))
                        confidence = float(smoothed_preds[best_idx])
                        bird_name = classes[best_idx]

                        duration = int((time.time() - start_time) * 1000)
                        
                        ignore_words = ['cricket', 'frog', 'toad', 'cicada', 'bat', 'grasshopper', 'insect', 'noise', 'locust', 'katydid', 'roesel', 'shrew', 'mouse']
                        is_ignored = any(w in bird_name.lower() for w in ignore_words)

                        if confidence > 0.5 and not is_ignored:
                            current_time = time.time()
                            is_new_encounter = True
                            
                            if last_seen_state["bird"] == bird_name and (current_time - last_seen_state["time"]) < 10.0:
                                is_new_encounter = False
                                
                            last_seen_state["bird"] = bird_name
                            last_seen_state["time"] = current_time

                            pred_str = f"{bird_name} ({confidence*100:.1f}%)"
                            esp32_data[ip]["prediction"] = pred_str
                            
                            # Save Audio Snippet if confidence > 85%
                            audio_filename = None
                            if confidence >= 0.85:
                                clean_name = bird_name.replace(" ", "_").replace("/", "-")
                                ts = time.strftime("%H-%M-%S")
                                audio_filename = f"{active_date}_{ts}_{clean_name}.wav"
                                audio_path = os.path.join(AUDIO_DIR, audio_filename)
                                audio_final_int16 = (audio_float * 32767.0).astype(np.int16)
                                wavfile.write(audio_path, SAMPLE_RATE, audio_final_int16)
                            
                            log_detection(ip, bird_name, confidence * 100, audio_filename)
                            print(f"[{ip}] {pred_str} | {duration}ms")

                            if is_new_encounter:
                                lifetime_stats[bird_name] += 1
                                save_lifetime_stats()

                    except Exception as ai_e:
                        print(f"[{ip}] AI Error: {ai_e}")

    except Exception as e:
        print(f"[-] Connection {ip} lost: {e}")
    finally:
        if ip in esp32_data: del esp32_data[ip]
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

# --- FASTAPI WEBSERVER ---
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
        :root { --primary: #2c3e50; --secondary: #00A2E8; --bg: #ecf0f1; --card: #ffffff; --accent: #FF7F29;}
        body { font-family: 'Inter', sans-serif; background-color: var(--bg); margin: 0; padding: 0; }
        header { background-color: #ffffff; color: var(--primary); padding: 15px 30px; display: flex; align-items: center; gap: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-bottom: 3px solid var(--secondary); }
        .logo { height: 60px; object-fit: contain; }
        .tabs { display: flex; background: var(--primary); padding: 0 20px; }
        .tab-btn { background: none; border: none; color: #bdc3c7; padding: 12px 24px; font-size: 16px; cursor: pointer; transition: 0.3s; font-family: 'Inter', sans-serif; }
        .tab-btn:hover { color: white; }
        .tab-btn.active { color: white; border-bottom: 3px solid var(--secondary); font-weight: 600; }
        .tab-content { display: none; padding: 30px; max-width: 1200px; margin: 0 auto; }
        .tab-content.active { display: block; }
        .card { background: var(--card); padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .bar-container { width: 100%; background-color: #e0e0e0; border-radius: 6px; height: 20px; margin-top: 10px;}
        .bar { height: 100%; background-color: var(--accent); width: 0%; transition: width 0.1s ease; border-radius: 6px; }
        .bar.active { background-color: #e74c3c; }
        .prediction { font-size: 18px; color: var(--accent); font-weight: 700; margin-top: 10px;}
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: var(--primary); color: white; font-weight: 600; }
        .chart-container { position: relative; height: 40vh; width: 100%; margin-top: 20px;}
        .config-row { margin-bottom: 20px; }
        .controls { display: flex; gap: 15px; align-items: center; margin-bottom: 15px; flex-wrap: wrap; }
        .date-picker, .btn { padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-family: 'Inter'; font-size: 14px; }
        .btn { background: var(--primary); color: white; cursor: pointer; border: none; font-weight: 600; }
        .btn:hover { background: var(--secondary); }
        input[type="range"] { accent-color: var(--accent); }
        audio { height: 30px; }
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

    <!-- 1. Day Overview (History) -->
    <div id="History" class="tab-content active">
        <h2>Day Overview Map</h2>
        <div class="card">
            <div class="controls">
                <input type="date" id="history-date" class="date-picker" onchange="fetchDayHistory()">
                <label><input type="checkbox" id="toggle-activity" checked onchange="toggleChartElements()"> Show Overall Activity</label>
                <label><input type="checkbox" id="toggle-temp" onchange="toggleWeather()"> Show Temperature</label>
                
                <select id="export-range" class="date-picker" style="margin-left: auto;">
                    <option value="today">Export Selected Day</option>
                    <option value="7days">Export Last 7 Days</option>
                    <option value="all">Export All History</option>
                </select>
                <button class="btn" onclick="exportCSV()">Download CSV</button>
            </div>
            <div class="chart-container" style="height: 50vh;">
                <canvas id="timelineChart"></canvas>
            </div>
            <table>
                <thead><tr><th>Time</th><th>Species</th><th>Confidence</th><th>Audio Replay</th></tr></thead>
                <tbody id="history-table"><tr><td colspan='4'>No data available</td></tr></tbody>
            </table>
        </div>
    </div>

    <!-- 2. Lifetime Stats -->
    <div id="Stats" class="tab-content">
        <h2>Total Detections (Lifetime)</h2>
        <div class="card">
            <div class="chart-container">
                <canvas id="speciesChart"></canvas>
            </div>
        </div>
    </div>

    <!-- 3. Live Radar -->
    <div id="Live" class="tab-content">
        <h2>Active Audio Streams</h2>
        <div id="nodes"><i>Waiting for ESP32 connections...</i></div>
    </div>

    <!-- 4. Settings -->
    <div id="AI" class="tab-content">
        <div class="card">
            <h3>AI Settings</h3>
            <div class="config-row">
                <label><b>Always-On Mode:</b></label><br>
                <input type="checkbox" id="always_on" onchange="sendConfig()"> Enable continuous inference
            </div>
            <div class="config-row">
                <label><b>RMS Threshold (Wake-on-Sound):</b> <span id="thresh-val">0.05</span></label><br>
                <input type="range" id="threshold" min="0.001" max="0.3" step="0.005" value="0.05" style="width: 300px;" onchange="sendConfig()" oninput="document.getElementById('thresh-val').innerText=this.value">
            </div>
            <div class="config-row">
                <label><b>Temporal Smoothing Factor:</b> <span id="smooth-val">0.5</span></label><br>
                <input type="range" id="smoothing" min="0" max="1" step="0.1" value="0.5" style="width: 300px;" onchange="sendConfig()" oninput="document.getElementById('smooth-val').innerText=this.value">
            </div>
            <hr>
            <h3>Storage & Location</h3>
            <div class="config-row">
                <label><b>History Retention (Days):</b> <span id="retention-val">7</span></label><br>
                <input type="number" id="retention" min="1" max="365" value="7" style="width: 100px; padding: 5px;" onchange="sendConfig()">
            </div>
            <div class="config-row">
                <label><b>Latitude (for Weather):</b></label>
                <input type="number" id="lat" value="49.4521" step="0.0001" style="width: 100px; padding: 5px;" onchange="sendConfig()">
                <label style="margin-left: 10px;"><b>Longitude:</b></label>
                <input type="number" id="lon" value="11.0767" step="0.0001" style="width: 100px; padding: 5px;" onchange="sendConfig()">
            </div>
        </div>
    </div>

    <script>
        let configLoaded = false;
        let currentHistoryLength = -1;
        let currentRenderedDate = "";
        let weatherDataCache = {}; 
        
        const todayStr = new Date().toISOString().split('T')[0];
        document.getElementById('history-date').value = todayStr;

        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) tabcontent[i].style.display = "none";
            tablinks = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tablinks.length; i++) tablinks[i].className = tablinks[i].className.replace(" active", "");
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }

        var ws = new WebSocket("ws://" + location.host + "/ws");
        var chartInstance = null;
        var timelineChart = null;
        
        const colorPalette = ['#00A2E8', '#FF7F29', '#E74C3C', '#2ECC71', '#9B59B6', '#F1C40F', '#34495E', '#E67E22', '#16A085'];
        let birdColorMap = {};

        function getBirdColor(birdName, index) {
            if (!birdColorMap[birdName]) {
                birdColorMap[birdName] = colorPalette[Object.keys(birdColorMap).length % colorPalette.length];
            }
            return birdColorMap[birdName];
        }

        function sendConfig() {
            ws.send(JSON.stringify({
                type: "config",
                always_on: document.getElementById('always_on').checked,
                threshold: parseFloat(document.getElementById('threshold').value),
                smoothing: parseFloat(document.getElementById('smoothing').value),
                retention_days: parseInt(document.getElementById('retention').value),
                latitude: parseFloat(document.getElementById('lat').value),
                longitude: parseFloat(document.getElementById('lon').value)
            }));
        }
        
        function exportCSV() {
            let range = document.getElementById('export-range').value;
            if(range === 'today') range = document.getElementById('history-date').value;
            window.location.href = "/api/export?range=" + range;
        }

        async function fetchDayHistory() {
            const dateStr = document.getElementById('history-date').value;
            const res = await fetch('/api/history/' + dateStr);
            const data = await res.json();
            
            if(document.getElementById('toggle-temp').checked) {
                await fetchWeatherData(dateStr);
            }
            
            renderDayData(data, dateStr, true); 
        }
        
        async function fetchWeatherData(dateStr) {
            if (weatherDataCache[dateStr]) return;
            const lat = document.getElementById('lat').value;
            const lon = document.getElementById('lon').value;
            try {
                // Try forecast API first
                let res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m&start_date=${dateStr}&end_date=${dateStr}`);
                let json = await res.json();
                if(json.error) throw new Error("Need archive");
                weatherDataCache[dateStr] = json.hourly.temperature_2m;
            } catch(e) {
                // Fallback to archive API for past dates
                let res = await fetch(`https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&hourly=temperature_2m&start_date=${dateStr}&end_date=${dateStr}`);
                let json = await res.json();
                if(!json.error) weatherDataCache[dateStr] = json.hourly.temperature_2m;
            }
        }
        
        async function toggleWeather() {
            if (document.getElementById('toggle-temp').checked) {
                const dateStr = document.getElementById('history-date').value;
                await fetchWeatherData(dateStr);
            }
            toggleChartElements();
        }

        function toggleChartElements() {
            if(!timelineChart) return;
            const showAct = document.getElementById('toggle-activity').checked;
            const showTemp = document.getElementById('toggle-temp').checked;
            
            timelineChart.data.datasets.forEach(ds => {
                if(ds.label === 'Overall Activity') ds.hidden = !showAct;
                if(ds.label === 'Temperature (°C)') ds.hidden = !showTemp;
            });
            timelineChart.update('none');
        }

        function updateStatsChart(statsData) {
            var ctx = document.getElementById('speciesChart').getContext('2d');
            var labels = Object.keys(statsData);
            var data = Object.values(statsData);
            var bgColors = labels.map((b, i) => getBirdColor(b, i));

            if(chartInstance) {
                chartInstance.data.labels = labels;
                chartInstance.data.datasets[0].data = data;
                chartInstance.data.datasets[0].backgroundColor = bgColors;
                chartInstance.update('none'); 
            } else {
                chartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: { labels: labels, datasets: [{ data: data, backgroundColor: bgColors }] },
                    options: { 
                        responsive: true, 
                        maintainAspectRatio: false, 
                        animation: false,
                        plugins: { legend: { display: false } }, // Fix the blue box issue
                        scales: { y: { beginAtZero: true, ticks: {stepSize: 1} } } 
                    }
                });
            }
        }

        function renderDayData(historyData, dateStr, forceRedraw = false) {
            if (!forceRedraw && historyData.length === currentHistoryLength && dateStr === currentRenderedDate) {
                return; 
            }
            currentHistoryLength = historyData.length;
            currentRenderedDate = dateStr;

            var ctxT = document.getElementById('timelineChart').getContext('2d');
            var datasets = [];
            var labelsY = [];
            var groupedByBird = {};
            
            let hourlyCounts = new Array(24).fill(0);

            historyData.forEach(function(item) {
                if (!labelsY.includes(item.bird)) labelsY.push(item.bird);
                if (!groupedByBird[item.bird]) groupedByBird[item.bird] = [];
                let fullIsoString = dateStr + 'T' + item.time;
                groupedByBird[item.bird].push({ x: fullIsoString, y: item.bird });
                let hour = parseInt(item.time.split(':')[0]);
                hourlyCounts[hour]++;
            });
            
            // 1. Overall Activity Background
            let activityData = [];
            for(let i=0; i<24; i++) {
                let h = i.toString().padStart(2, '0');
                activityData.push({ x: `${dateStr}T${h}:30:00`, y: hourlyCounts[i] });
            }
            datasets.push({
                type: 'line',
                label: 'Overall Activity',
                data: activityData,
                yAxisID: 'yActivity',
                borderColor: 'rgba(0,0,0,0.15)',
                backgroundColor: 'rgba(0,0,0,0.05)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                hidden: !document.getElementById('toggle-activity').checked
            });
            
            // 2. Weather Data (if fetched)
            if(weatherDataCache[dateStr]) {
                let tempData = [];
                for(let i=0; i<24; i++) {
                    let h = i.toString().padStart(2, '0');
                    tempData.push({ x: `${dateStr}T${h}:30:00`, y: weatherDataCache[dateStr][i] });
                }
                datasets.push({
                    type: 'line',
                    label: 'Temperature (°C)',
                    data: tempData,
                    yAxisID: 'yTemp',
                    borderColor: '#e74c3c',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    hidden: !document.getElementById('toggle-temp').checked
                });
            }

            // 3. Bird Scatters
            for (var bird in groupedByBird) {
                datasets.push({
                    type: 'scatter',
                    label: bird,
                    data: groupedByBird[bird],
                    yAxisID: 'y',
                    backgroundColor: getBirdColor(bird),
                    borderColor: getBirdColor(bird),
                    showLine: true,       
                    borderWidth: 4,       
                    pointRadius: 5,       
                    pointHoverRadius: 8
                });
            }

            if(timelineChart) {
                timelineChart.data.datasets = datasets;
                timelineChart.options.scales.x.min = dateStr + 'T00:00:00';
                timelineChart.options.scales.x.max = dateStr + 'T23:59:59';
                timelineChart.options.scales.y.labels = labelsY;
                timelineChart.update('none'); 
            } else {
                timelineChart = new Chart(ctxT, {
                    data: { datasets: datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: false, 
                        scales: {
                            x: { 
                                type: 'time', 
                                time: { unit: 'hour', displayFormats: { hour: 'HH:mm' } },
                                min: dateStr + 'T00:00:00',
                                max: dateStr + 'T23:59:59',
                                title: {display: true, text: 'Time of Day', font: {family: 'Inter', weight: 'bold'}} 
                            },
                            y: { 
                                type: 'category', 
                                labels: labelsY, 
                                position: 'left',
                                grid: { color: 'rgba(0,0,0,0.05)' },
                                title: {display: true, text: 'Species', font: {family: 'Inter', weight: 'bold'}} 
                            },
                            yActivity: { type: 'linear', position: 'right', display: false, min: 0 },
                            yTemp: { type: 'linear', position: 'right', display: false }
                        },
                        plugins: {
                            legend: { display: false }, 
                            tooltip: { 
                                callbacks: { 
                                    label: function(context) { 
                                        if (context.dataset.label === 'Overall Activity') return context.raw.y + ' calls this hour';
                                        if (context.dataset.label === 'Temperature (°C)') return context.raw.y + ' °C';
                                        let t = new Date(context.raw.x).toLocaleTimeString();
                                        return context.raw.y + " at " + t; 
                                    } 
                                } 
                            }
                        }
                    }
                });
            }

            var histTbody = document.getElementById("history-table");
            histTbody.innerHTML = "";
            if (historyData.length === 0) {
                histTbody.innerHTML = "<tr><td colspan='4'>No data available</td></tr>";
            } else {
                let reverseData = [...historyData].reverse();
                reverseData.forEach(function(item) {
                    var tr = document.createElement("tr");
                    let audioHtml = item.audio ? `<audio controls src="${item.audio}"></audio>` : `<span style="color:#aaa;">---</span>`;
                    tr.innerHTML = `<td>${item.time}</td><td>${item.bird}</td><td>${item.conf.toFixed(1)}%</td><td>${audioHtml}</td>`;
                    histTbody.appendChild(tr);
                });
            }
        }

        ws.onmessage = function(event) {
            var payload = JSON.parse(event.data);

            if (!configLoaded && payload.config) {
                document.getElementById('always_on').checked = payload.config.always_on;
                document.getElementById('threshold').value = payload.config.threshold;
                document.getElementById('thresh-val').innerText = payload.config.threshold;
                document.getElementById('smoothing').value = payload.config.smoothing;
                document.getElementById('smooth-val').innerText = payload.config.smoothing;
                if(payload.config.retention_days) {
                    document.getElementById('retention').value = payload.config.retention_days;
                    document.getElementById('retention-val').innerText = payload.config.retention_days;
                }
                if(payload.config.latitude) document.getElementById('lat').value = payload.config.latitude;
                if(payload.config.longitude) document.getElementById('lon').value = payload.config.longitude;
                configLoaded = true;
                
                fetchDayHistory();
            }

            var nodesDiv = document.getElementById("nodes");
            if (Object.keys(payload.nodes).length === 0) nodesDiv.innerHTML = "<i>Waiting for ESP32...</i>";
            else {
                nodesDiv.innerHTML = "";
                for (var ip in payload.nodes) {
                    var node = payload.nodes[ip];
                    var percent = Math.min(100, (node.rms / 0.2) * 100);
                    var barClass = (node.rms > payload.config.threshold || payload.config.always_on) ? "bar active" : "bar";
                    var card = document.createElement("div");
                    card.className = "card";
                    card.innerHTML = "<h3>Node IP: " + ip + "</h3>" +
                                     "<p><b>Status:</b> " + node.status + " | <b>RMS:</b> " + node.rms.toFixed(3) + "</p>" +
                                     "<div class='bar-container'><div class='" + barClass + "' style='width: " + percent + "%'></div></div>" +
                                     "<div class='prediction'>Last Detection: " + node.prediction + "</div>";
                    nodesDiv.appendChild(card);
                }
            }

            let selectedDate = document.getElementById('history-date').value;
            if (selectedDate === new Date().toISOString().split('T')[0] && payload.today_data) {
                renderDayData(payload.today_data, selectedDate, false);
            }

            if(Object.keys(payload.stats).length > 0) updateStatsChart(payload.stats);
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
async def export_data(range: str = 'today'):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Time', 'Bird Species', 'Confidence', 'Audio Link'])
    
    files_to_read = []
    if range == 'all':
        files_to_read = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    elif range == '7days':
        now = time.time()
        for f in glob.glob(os.path.join(HISTORY_DIR, "*.json")):
            if os.stat(f).st_mtime >= now - (7 * 86400):
                files_to_read.append(f)
    else:
        # specific date string e.g. "2026-04-11"
        target_file = os.path.join(HISTORY_DIR, f"{range}.json")
        if os.path.exists(target_file):
            files_to_read.append(target_file)

    for file_path in sorted(files_to_read):
        date_str = os.path.basename(file_path).replace(".json", "")
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                for item in data:
                    writer.writerow([date_str, item.get("time",""), item.get("bird",""), item.get("conf",""), item.get("audio", "")])
            except: pass

    filename = f"bird_data_{range}.csv"
    return PlainTextResponse(
        output.getvalue(), 
        media_type="text/csv", 
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                if data.get("type") == "config":
                    current_config["always_on"] = data.get("always_on", False)
                    current_config["threshold"] = data.get("threshold", 0.05)
                    current_config["smoothing"] = data.get("smoothing", 0.5)
                    current_config["retention_days"] = data.get("retention_days", 7)
                    if "latitude" in data: current_config["latitude"] = data["latitude"]
                    if "longitude" in data: current_config["longitude"] = data["longitude"]
                    save_config() 
            except asyncio.TimeoutError:
                pass

            payload = {
                "nodes": esp32_data,
                "today_data": active_day_data,
                "config": current_config,
                "stats": dict(lifetime_stats)
            }
            await websocket.send_json(payload)

    except Exception:
        pass

if __name__ == "__main__":
    threading.Thread(target=start_socket_server, daemon=True).start()
    print("[*] Starting Web-Dashboard on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
