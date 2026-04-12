import socket
import threading
import numpy as np
import librosa
import onnxruntime as ort
import asyncio
import os
import json
import time
import torch
import torch.nn.functional as F
import scipy.signal
from collections import deque, Counter
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- KONFIGURATION ---
HOST = '0.0.0.0'
PORT = 5000
SAMPLE_RATE = 32000
DURATION = 5
GAIN = 30.0  

# Bandpass Filter für Vögel (1kHz - 12kHz)
b_filter, a_filter = scipy.signal.butter(4, [1000, 12000], btype='bandpass', fs=SAMPLE_RATE)

# Pfade
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")

current_config = {
    "threshold": 0.05,
    "always_on": False,
    "smoothing": 0.5,
    "model_file": "efficientnet_fp32.onnx",
    "classes_file": "efficientnet_task-species_ch-3_attn-False_layers-3_classes.json"
}

# --- ZUSTAND & DATENBANK ---
esp32_data = {}
history_data = {}
lifetime_stats = Counter()

DB_FILE = "lifetime_stats.json"
if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, "r") as f:
            lifetime_stats.update(json.load(f))
    except Exception: pass

def save_lifetime_stats():
    with open(DB_FILE, "w") as f:
        json.dump(dict(lifetime_stats), f)

# --- KI MODELL LADEN ---
session = None
input_name = None
classes = []

def load_ai_model():
    global session, input_name, classes
    model_path = os.path.join(MODELS_DIR, current_config["model_file"])
    classes_path = os.path.join(MODELS_DIR, current_config["classes_file"])
    try:
        print(f"[*] Lade KI Modell: {model_path}...")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name

        with open(classes_path, "r") as f:
            data = json.load(f)
            classes = data["classes"] if isinstance(data, dict) and "classes" in data else data

        print(f"[+] Modell erfolgreich geladen. ({len(classes)} Klassen)")
    except Exception as e:
        print(f"[-] Fehler beim Laden des Modells: {e}")

load_ai_model()

# --- AUDIO VORVERARBEITUNG ---
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
    print(f"\n[+] ESP32 {ip} verbunden!")
    esp32_data[ip] = {"rms": 0.0, "status": "Lauscht...", "prediction": "-"}
    history_data[ip] = deque(maxlen=50)
    
    # Mono: 32000 Samples * 1 Kanal * 2 Bytes
    BYTES_PER_SECOND = SAMPLE_RATE * 2 
    BUFFER_MAX = BYTES_PER_SECOND * DURATION 
    raw_buffer = bytearray()
    last_preds = deque(maxlen=3)
    
    bytes_since_last_process = 0
    PROCESS_EVERY_BYTES = BYTES_PER_SECOND # Inferenz alle 1 Sekunde

    try:
        while True:
            data = conn.recv(4096)
            if not data: break
            
            raw_buffer.extend(data)
            bytes_since_last_process += len(data)

            if len(raw_buffer) > BUFFER_MAX:
                raw_buffer = raw_buffer[-BUFFER_MAX:]

            # UI Update (RMS)
            ui_chunk = BYTES_PER_SECOND
            if len(raw_buffer) >= ui_chunk:
                mono_audio = np.frombuffer(bytes(raw_buffer[-ui_chunk:]), dtype=np.int16)
                
                audio_float_ui = (mono_audio.astype(np.float32) / 32768.0) * GAIN
                audio_float_ui = scipy.signal.filtfilt(b_filter, a_filter, audio_float_ui)
                audio_float_ui = np.clip(audio_float_ui, -1.0, 1.0)
                
                rms = float(np.sqrt(np.mean(audio_float_ui**2)))
                esp32_data[ip]["rms"] = rms

                if rms > current_config["threshold"] or current_config["always_on"]:
                    esp32_data[ip]["status"] = "Aufnahme & KI-Inferenz läuft..."
                else:
                    esp32_data[ip]["status"] = "Lauscht auf Geräusche..."

            # KI Inferenz
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
                        timestamp = time.strftime("%H:%M:%S")

                        ignore_words = ['cricket', 'frog', 'toad', 'cicada', 'bat', 'grasshopper', 'insect', 'noise', 'locust', 'katydid', 'roesel', 'shrew', 'mouse']
                        is_ignored = any(w in bird_name.lower() for w in ignore_words)

                        if confidence > 0.5 and not is_ignored:
                            pred_str = f"{bird_name} ({confidence*100:.1f}%)"
                            esp32_data[ip]["prediction"] = pred_str

                            history_data[ip].append({"time": timestamp, "bird": bird_name, "conf": confidence*100})
                            lifetime_stats[bird_name] += 1
                            save_lifetime_stats()

                            print(f"[{ip}] {timestamp} | {pred_str} | {duration}ms")

                    except Exception as ai_e:
                        print(f"[{ip}] KI Fehler: {ai_e}")

    except Exception as e:
        print(f"[-] Verbindung {ip} getrennt: {e}")
    finally:
        if ip in esp32_data: del esp32_data[ip]
        if ip in history_data: del history_data[ip]
        conn.close()

def start_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[*] I2S Socket Server lauscht auf {HOST}:{PORT}")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

# --- FASTAPI WEBSERVER ---
os.makedirs("static", exist_ok=True)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

html_content = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>Piepmatz Polizei Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { --primary: #2c3e50; --secondary: #18bc9c; --bg: #ecf0f1; --card: #ffffff; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background-color: var(--bg); margin: 0; padding: 0; }
        header { background-color: #ffffff; color: var(--primary); padding: 15px 30px; display: flex; align-items: center; gap: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-bottom: 3px solid var(--secondary); }
        .logo { height: 60px; object-fit: contain; }
        .tabs { display: flex; background: var(--primary); padding: 0 20px; }
        .tab-btn { background: none; border: none; color: #bdc3c7; padding: 12px 24px; font-size: 16px; cursor: pointer; transition: 0.3s; }
        .tab-btn:hover { color: white; }
        .tab-btn.active { color: white; border-bottom: 3px solid var(--secondary); font-weight: bold; }
        .tab-content { display: none; padding: 30px; max-width: 1200px; margin: 0 auto; }
        .tab-content.active { display: block; }
        .card { background: var(--card); padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .bar-container { width: 100%; background-color: #e0e0e0; border-radius: 6px; height: 20px; margin-top: 10px;}
        .bar { height: 100%; background-color: var(--secondary); width: 0%; transition: width 0.1s ease; }
        .bar.active { background-color: #e74c3c; }
        .prediction { font-size: 18px; color: #e67e22; font-weight: bold; margin-top: 10px;}
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: var(--primary); color: white; }
        .chart-container { position: relative; height: 40vh; width: 100%; margin-top: 20px;}
        .config-row { margin-bottom: 20px; }
    </style>
</head>
<body>
    <header>
        <img src="/static/logo.png" alt="Logo" class="logo" onerror="this.style.display='none'">
        <h1>Piepmatz Polizei - KI Monitor</h1>
    </header>

    <div class="tabs">
        <button class="tab-btn active" onclick="openTab(event, 'Live')">Live-Radar</button>
        <button class="tab-btn" onclick="openTab(event, 'History')">Aktivitäts-Timeline</button>
        <button class="tab-btn" onclick="openTab(event, 'Stats')">Lebenszeit Analyse</button>
        <button class="tab-btn" onclick="openTab(event, 'AI')">Einstellungen</button>
    </div>

    <div id="Live" class="tab-content active">
        <h2>Aktuelle Audio-Streams</h2>
        <div id="nodes"><i>Warte auf ESP32 Verbindungen...</i></div>
    </div>

    <div id="History" class="tab-content">
        <h2>Zeitliche Erkennungs-Map</h2>
        <div class="card">
            <div class="chart-container" style="height: 50vh;">
                <canvas id="timelineChart"></canvas>
            </div>
            <table>
                <thead><tr><th>Uhrzeit</th><th>Spezies</th><th>Konfidenz</th></tr></thead>
                <tbody id="history-table"><tr><td colspan='3'>Noch keine Daten</td></tr></tbody>
            </table>
        </div>
    </div>

    <div id="Stats" class="tab-content">
        <h2>Gesamte Erkennungen (Lifetime)</h2>
        <div class="card">
            <div class="chart-container">
                <canvas id="speciesChart"></canvas>
            </div>
        </div>
    </div>

    <div id="AI" class="tab-content">
        <div class="card">
            <h3>KI Einstellungen</h3>
            <div class="config-row">
                <label><b>Always-On Modus:</b></label><br>
                <input type="checkbox" id="always_on" onchange="sendConfig()"> Aktivieren
            </div>
            <div class="config-row">
                <label><b>RMS Schwellenwert (Audio-Lautstärke):</b> <span id="thresh-val">0.05</span></label><br>
                <input type="range" id="threshold" min="0.001" max="0.3" step="0.005" value="0.05" style="width: 300px;" onchange="sendConfig()" oninput="document.getElementById('thresh-val').innerText=this.value">
            </div>
            <div class="config-row">
                <label><b>Glättungs-Faktor (Smoothing):</b> <span id="smooth-val">0.5</span></label><br>
                <input type="range" id="smoothing" min="0" max="1" step="0.1" value="0.5" style="width: 300px;" onchange="sendConfig()" oninput="document.getElementById('smooth-val').innerText=this.value">
            </div>
        </div>
    </div>

    <script>
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

        function sendConfig() {
            ws.send(JSON.stringify({
                type: "config",
                always_on: document.getElementById('always_on').checked,
                threshold: parseFloat(document.getElementById('threshold').value),
                smoothing: parseFloat(document.getElementById('smoothing').value)
            }));
        }

        function updateStatsChart(statsData) {
            var ctx = document.getElementById('speciesChart').getContext('2d');
            var labels = Object.keys(statsData);
            var data = Object.values(statsData);
            if(chartInstance) {
                chartInstance.data.labels = labels;
                chartInstance.data.datasets[0].data = data;
                chartInstance.update();
            } else {
                chartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: { labels: labels, datasets: [{ label: 'Anzahl Erkennungen', data: data, backgroundColor: '#18bc9c' }] },
                    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, ticks: {stepSize: 1} } } }
                });
            }
        }

        function updateTimelineChart(historyData) {
            var ctxT = document.getElementById('timelineChart').getContext('2d');
            var dataPoints = [];
            var labelsY = [];

            for (var ip in historyData) {
                historyData[ip].forEach(function(item) {
                    if (!labelsY.includes(item.bird)) labelsY.push(item.bird);
                    dataPoints.push({ x: item.time, y: item.bird });
                });
            }

            if(timelineChart) {
                timelineChart.data.datasets[0].data = dataPoints;
                timelineChart.options.scales.y.labels = labelsY;
                timelineChart.update();
            } else {
                timelineChart = new Chart(ctxT, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Vogel-Sichtungen',
                            data: dataPoints,
                            backgroundColor: 'rgba(24, 188, 156, 0.8)',
                            pointStyle: 'rectRounded',
                            radius: 12,
                            hitRadius: 15
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { type: 'category', title: {display: true, text: 'Uhrzeit'} },
                            y: { type: 'category', labels: labelsY, title: {display: true, text: 'Spezies'} }
                        },
                        plugins: {
                            tooltip: { callbacks: { label: function(context) { return context.raw.y + " um " + context.raw.x; } } }
                        }
                    }
                });
            }
        }

        ws.onmessage = function(event) {
            var payload = JSON.parse(event.data);

            var nodesDiv = document.getElementById("nodes");
            if (Object.keys(payload.nodes).length === 0) nodesDiv.innerHTML = "<i>Warte auf ESP32...</i>";
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
                                     "<div class='prediction'>Letzte Erkennung: " + node.prediction + "</div>";
                    nodesDiv.appendChild(card);
                }
            }

            var histTbody = document.getElementById("history-table");
            histTbody.innerHTML = "";
            var hasHistory = false;
            for (var ip in payload.history) {
                var items = [...payload.history[ip]].reverse();
                items.forEach(function(item) {
                    hasHistory = true;
                    var tr = document.createElement("tr");
                    tr.innerHTML = "<td>" + item.time + "</td><td>" + item.bird + "</td><td>" + item.conf.toFixed(1) + "%</td>";
                    histTbody.appendChild(tr);
                });
            }
            if(!hasHistory) histTbody.innerHTML = "<tr><td colspan='3'>Noch keine Daten</td></tr>";

            updateTimelineChart(payload.history);
            if(Object.keys(payload.stats).length > 0) updateStatsChart(payload.stats);
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                if data.get("type") == "config":
                    current_config["always_on"] = data.get("always_on", False)
                    current_config["threshold"] = data.get("threshold", 0.05)
                    current_config["smoothing"] = data.get("smoothing", 0.5)
            except asyncio.TimeoutError:
                pass

            payload = {
                "nodes": esp32_data,
                "history": {ip: list(hist) for ip, hist in history_data.items()},
                "config": current_config,
                "stats": dict(lifetime_stats)
            }
            await websocket.send_json(payload)

    except Exception:
        pass

if __name__ == "__main__":
    threading.Thread(target=start_socket_server, daemon=True).start()
    print("[*] Starte Web-Dashboard auf http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
