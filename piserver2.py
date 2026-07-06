import asyncio
import websockets
import json
import time
import math
import http.server
import threading
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8765
SEND_HZ = 20

REP_ANGLE_PEAK = 55
REP_RESET_ANGLE = 40

# ─── SHARED STATE ─────────────────────────────────────────────────────────────
current_mode: int = 2
_start_time = time.time()

# ─── SENSOR FUNCTIONS ─────────────────────────────────────────────────────────
def read_emg_quad() -> float:
    t = time.time()
    v = max(0, math.sin(t * math.pi * 0.6)) * 0.8
    noise = (abs(hash(int(t * 137))) % 100) / 1000
    return round(min(1.0, v + noise), 4)

def read_emg_ham() -> float:
    t = time.time()
    v = max(0, -math.sin(t * math.pi * 0.6)) * 0.55
    noise = (abs(hash(int(t * 97 + 1))) % 100) / 2000
    return round(min(1.0, v + noise), 4)

def read_knee_angle() -> float:
    t = time.time()
    wave = max(0, math.sin(t * math.pi * 0.6))
    noise = ((abs(hash(int(t * 200))) % 30) / 30 - 0.5)
    return round(wave * 75 + noise, 2)

def read_battery() -> float:
    elapsed = max(0, time.time() - _start_time)
    return round(max(0.05, 1.0 - elapsed * 0.00006), 3)

def read_mode_buttons() -> int:
    return current_mode

def compute_assist(mode: int, emg_quad: float, emg_ham: float) -> float:
    combined = emg_quad * 0.7 + emg_ham * 0.3
    gains = { 0: 0.0, 1: 0.15, 2: 0.35, 3: 0.65, 4: 0.50, 5: 0.40 }
    return round(min(1.0, combined * gains.get(mode, 0.35)), 3)

# ─── REP COUNTER ──────────────────────────────────────────────────────────────
class RepCounter:
    def __init__(self):
        self.count = 0
        self._peaked = False

    def update(self, angle: float) -> int:
        if angle >= REP_ANGLE_PEAK and not self._peaked:
            self._peaked = True
        elif angle < REP_RESET_ANGLE and self._peaked:
            self._peaked = False
            self.count += 1
            print(f"  Rep {self.count} counted  (angle back to {angle:.1f}°)")
        return self.count

# ─── WEBSOCKET HANDLER ────────────────────────────────────────────────────────
rep_counter = RepCounter()
connected_clients: set = set()

async def handle_client(websocket, path=None):
    global current_mode
    connected_clients.add(websocket)
    client = websocket.remote_address[0] if websocket.remote_address else "?"
    print(f"\n[+] Web app connected: {client}  (total clients: {len(connected_clients)})")

    async def send_loop():
        interval = 1.0 / SEND_HZ
        while True:
            t0 = time.time()
            angle    = read_knee_angle()
            emg_quad = read_emg_quad()
            emg_ham  = read_emg_ham()
            battery  = read_battery()
            mode     = read_mode_buttons()
            assist   = compute_assist(mode, emg_quad, emg_ham)
            reps     = rep_counter.update(angle)
            payload = {
                "angle":     angle,
                "emg_quad":  emg_quad,
                "emg_ham":   emg_ham,
                "mode":      mode,
                "reps":      reps,
                "battery":   battery,
                "assist":    assist,
                "timestamp": round(time.time(), 3),
            }
            await websocket.send(json.dumps(payload))
            elapsed = time.time() - t0
            await asyncio.sleep(max(0, interval - elapsed))

    async def recv_loop():
        global current_mode
        async for raw_msg in websocket:
            try:
                cmd = json.loads(raw_msg)
                command = cmd.get("command")
                if command == "set_mode":
                    new_mode = int(cmd["value"])
                    if 0 <= new_mode <= 5:
                        current_mode = new_mode
                        mode_names = {
                            0:"Passive", 1:"Low", 2:"Medium",
                            3:"High",   4:"Adaptive", 5:"Resistive"
                        }
                        print(f"  Mode changed → {mode_names.get(new_mode, new_mode)} (by web app)")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"  Bad command received: {e}")

    try:
        await asyncio.gather(send_loop(), recv_loop())
    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Web app disconnected: {client}")
    finally:
        connected_clients.discard(websocket)

# ─── HTTP SERVER ──────────────────────────────────────────────────────────────
def start_http_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("0.0.0.0", 8080), handler)
    print("  Web UI  : http://0.0.0.0:8080")
    httpd.serve_forever()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 52)
    print("  RAF Exoskeleton — WebSocket Server  (v2)")
    print("=" * 52)
    print(f"  Host : {HOST}:{PORT}")
    print(f"  Rate : {SEND_HZ} Hz")
    print()
    print("  1. Find your Pi IP:   hostname -I")
    print("  2. Open in ANY browser:")
    print("     http://<Pi-IP>:8080/RAF_Monitor_v2.html")
    print("=" * 52)

    t = threading.Thread(target=start_http_server, daemon=True)
    t.start()

    async with websockets.serve(handle_client, HOST, PORT, ping_interval=None):
        print(f"\n  Listening on ws://0.0.0.0:{PORT} ...\n")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Server stopped.")


