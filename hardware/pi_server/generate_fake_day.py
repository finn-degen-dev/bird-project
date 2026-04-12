import json
import random
from datetime import datetime, timedelta
import os

# --- Configuration ---
DATE_STR = "2026-04-11"
OUTPUT_FILE = f"../../data/history/{DATE_STR}.json"

# Typical birds for a small town near Nuremberg in April
BIRDS = [
    "Eurasian Blackbird", "European Robin", "Great Tit", "Eurasian Blue Tit",
    "House Sparrow", "Common Chaffinch", "Common Wood Pigeon", 
    "Eurasian Collared Dove", "Common Chiffchaff", "Eurasian Jay"
]
NIGHT_BIRDS = ["Tawny Owl"]

# Create data/history/ directory if it doesn't exist
os.makedirs("data/history", exist_ok=True)

data = []
current_time = datetime.strptime(f"{DATE_STR} 00:00:00", "%Y-%m-%d %H:%M:%S")
end_time = current_time + timedelta(days=1)

print(f"Generating massive realistic dataset for {DATE_STR}...")

while current_time < end_time:
    hour = current_time.hour
    
    # Determine the probability of a bird starting to sing in this exact second
    # and pick the active birds for this time of day.
    if 4 <= hour < 5:      # Pre-dawn
        prob = 0.005
        active = ["Eurasian Blackbird", "European Robin"]
    elif 5 <= hour < 8:    # DAWN CHORUS (Extremely active)
        prob = 0.03
        active = BIRDS
    elif 8 <= hour < 12:   # Morning (Active)
        prob = 0.01
        active = BIRDS
    elif 12 <= hour < 17:  # Afternoon (Quiet)
        prob = 0.002
        active = ["House Sparrow", "Common Wood Pigeon", "Great Tit"]
    elif 17 <= hour < 20:  # Evening Chorus (Active again)
        prob = 0.015
        active = ["Eurasian Blackbird", "European Robin", "Common Chiffchaff"]
    else:                  # Night (Very quiet)
        prob = 0.0005
        active = NIGHT_BIRDS

    # Did a bird start a "singing session" right now?
    if random.random() < prob:
        bird = random.choice(active)
        
        # How long does this bird stay in the tree singing? (between 20 seconds and 5 minutes)
        session_duration = random.randint(20, 300) 
        session_end = current_time + timedelta(seconds=session_duration)
        
        # Generate dense data points while the bird is singing
        while current_time < session_end and current_time < end_time:
            # Model inference triggers every 1 to 4 seconds
            current_time += timedelta(seconds=random.randint(1, 4))
            
            # Confidence naturally fluctuates between 60% and 99%
            conf = round(random.uniform(60.0, 99.9), 1)
            
            data.append({
                "time": current_time.strftime("%H:%M:%S"),
                "bird": bird,
                "conf": conf,
                "ip": "192.168.1.104" # Fake ESP32 IP
            })
            
    # Move time forward by 1 second to check again
    current_time += timedelta(seconds=1)

# Sort chronologically just to be safe
data.sort(key=lambda x: x["time"])

# Save to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f)

print(f"Done! Generated {len(data)} individual continuous data points.")
print(f"Saved to {OUTPUT_FILE}. You can now refresh your dashboard!")
