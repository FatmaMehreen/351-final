import asyncio
import websockets
import json
import time
import math
import http.server
import threading
import os
import csv
import numpy as np
from collections import deque

try:
    import odrive
    from odrive.enums import *
    from mpu6050 import mpu6050
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[!] odrive / mpu6050 not importable on this machine -> High mode will report "
          "'NO HARDWARE' instead of running. Run this on the Pi with both libs installed.")

# ─── DASHBOARD / NETWORK CONFIG ───────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8765
SEND_HZ = 20

# Which dashboard mode tile triggers your real walking-assist script.
# Dashboard ids: 0 Passive, 1 Low, 2 Medium, 3 High, 4 Adaptive, 5 Resistive
WALKING_CODE_MODE_ID = 3   # "High" tile
MODE_NAMES = {0: "Passive", 1: "Low", 2: "Medium", 3: "High", 4: "Adaptive", 5: "Resistive"}

# Dashboard-only display thresholds (your walking script has no rep counter —
# these two lines are new, just so the "Reps" / "Peak" boxes on screen have
# something to show. They never feed back into the motor control below.)
DASH_REP_PEAK_ANGLE  = 55
DASH_REP_RESET_ANGLE = 40

# ════════════════════════════════════════════════════════════════════════
# EVERYTHING IN THIS SECTION IS YOUR WALKING SCRIPT.
# All constants, formulas, thresholds, and control logic below are exactly
# what you gave me. The ONLY mechanical changes, needed purely because this
# now lives inside a function instead of running at the top of its own
# file, are marked with "# <-- added" comments. Nothing about how the
# motor is driven was touched.
# ════════════════════════════════════════════════════════════════════════
MODE_NAME = "RAF EXOSKELETON — WALKING v5"

PITCH_SAFE_MIN   = -5.0    # hard safety cutoff — too straight
PITCH_SAFE_MAX   = -62.0   # hard safety cutoff — too bent

FLEX_MIN_PITCH   = -15.0   # must be at least this bent to get flex assist
EXT_MIN_PITCH    = -13.0   # must have been bent at least this much

TORQUE_FLEX_MAX  = 0.70     # Nm max flex assist
TORQUE_EXT_MAX   = 0.40    # Nm max extension assist
TORQUE_MIN       = 0.20   # Nm minimum to actually move motor

GZ_MAX           = 20.0    # deg/s at which max torque is reached

VELOCITY_LIM     = 6.0
GYRO_THRESHOLD   = 6      # deg/s — minimum gz to trigger
GYRO_HYSTERESIS  = 2

EXT_CONFIRM_TIME = 0.25    # seconds

STARTUP_LOCKOUT  = 5.0
CF_ALPHA         = 0.98

# module-level state (same names/roles as your original top-level script)
imu = None
odrv = None
axis = None
LOG_FILE = None
last_loop_time = time.time()
cf_angle = 0.0
last_imu = time.time()
GZ_BUFFER = deque([0.0, 0.0, 0.0], maxlen=3)
PITCH_BUFFER = deque([0.0, 0.0, 0.0, 0.0, 0.0], maxlen=5)
current_direction = 0
ext_candidate_since = None
motor_engaged = False
session_torques = []
start_time = 0.0
watchdog_fault = False                 # <-- added, dashboard status only
rep_count_walking = 0                  # <-- added, dashboard display only
rep_peaked_walking = False             # <-- added, dashboard display only

# telemetry bridge to the websocket sender — the only other addition.
# It just reads the same pitch/direction/torque/status values your loop
# already computes and copies them out so the dashboard has numbers to show.
latest_telemetry = {
    "angle": 0.0, "direction": 0, "torque": 0.0,
    "reps": 0, "status": "Select HIGH mode to start the walking-assist code",
    "running": False,
}
telemetry_lock = threading.Lock()

def update_telemetry(**kwargs):
    with telemetry_lock:
        latest_telemetry.update(kwargs)

def read_telemetry() -> dict:
    with telemetry_lock:
        return dict(latest_telemetry)

# ─── SENSOR / SAFETY / CONTROL FUNCTIONS (unchanged) ─────────────────────
def log_data(t, pitch, gz, gz_avg, trend, direction, torque, status):
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            round(t, 3), round(pitch, 2), round(gz, 2),
            round(gz_avg, 2), round(trend, 3),
            direction, round(torque, 4), status
        ])

def check_watchdog():
    global watchdog_fault
    if time.time() - last_loop_time > 1.0:
        axis.controller.input_torque = 0.0
        axis.requested_state         = AXIS_STATE_IDLE
        print("WATCHDOG TRIGGERED — motor cut")
        watchdog_fault = True          # <-- added, dashboard status only
        exit()

def get_accel_angle(accel):
    ax, ay, az = accel['x'], accel['y'], accel['z']
    return math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))

def update_imu():
    global cf_angle, last_imu
    now      = time.time()
    dt       = min(now - last_imu, 0.1)
    last_imu = now
    accel    = imu.get_accel_data()
    gyro     = imu.get_gyro_data()
    gz       = -gyro['z']
    cf_angle = (CF_ALPHA * (cf_angle + gz * dt) +
                (1 - CF_ALPHA) * get_accel_angle(accel))

    GZ_BUFFER.append(gz)
    PITCH_BUFFER.append(cf_angle)

    gz_avg      = sum(GZ_BUFFER) / len(GZ_BUFFER)
    pitch_trend = PITCH_BUFFER[-1] - PITCH_BUFFER[0]

    return cf_angle, gz, gz_avg, pitch_trend

def get_direction(gz_avg, pitch, pitch_trend):
    global current_direction, ext_candidate_since
    now = time.time()

    ON  = GYRO_THRESHOLD + GYRO_HYSTERESIS
    OFF = GYRO_THRESHOLD - GYRO_HYSTERESIS

    flex_signal = (gz_avg < -ON) and (pitch < FLEX_MIN_PITCH)

    ext_signal = (
        gz_avg > ON and
        pitch_trend > 0.1 and
        pitch < EXT_MIN_PITCH
    )

    if ext_signal:
        if ext_candidate_since is None:
            ext_candidate_since = now
        ext_confirmed = (now - ext_candidate_since) >= EXT_CONFIRM_TIME
    else:
        ext_candidate_since = None
        ext_confirmed = False

    if flex_signal:
        current_direction = -1
        ext_candidate_since = None

    elif ext_confirmed:
        current_direction = 1

    elif abs(gz_avg) < OFF:
        current_direction = 0
        ext_candidate_since = None

    return current_direction

def engage_motor():
    global motor_engaged
    if not motor_engaged:
        axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
        axis.controller.config.input_mode   = INPUT_MODE_PASSTHROUGH
        axis.requested_state                = AXIS_STATE_CLOSED_LOOP_CONTROL
        motor_engaged = True

def release_motor():
    global motor_engaged
    if motor_engaged:
        axis.controller.input_torque = 0.0
        axis.requested_state         = AXIS_STATE_IDLE
        motor_engaged = False

def calculate_torque(direction, gz, pitch, locked_out):
    if locked_out:
        return 0.0, False
    if pitch < PITCH_SAFE_MAX or pitch > PITCH_SAFE_MIN:
        return 0.0, False

    scale = min(abs(gz) / GZ_MAX, 1.0)

    if direction == -1:
        torque = -(TORQUE_FLEX_MAX * scale)
        if abs(torque) < TORQUE_MIN:
            torque = -TORQUE_MIN
        return torque, True

    elif direction == 1:
        torque = TORQUE_EXT_MAX * scale
        if torque < TORQUE_MIN:
            torque = TORQUE_MIN
        return torque, True

    return 0.0, False

def print_summary(duration):
    print("\n" + "="*45)
    print(f"  {MODE_NAME} SUMMARY")
    print("="*45)
    print(f"Duration   : {duration:.1f}s")
    if session_torques:
        active = [t for t in session_torques if abs(t) > 0.001]
        if active:
            print(f"Active time: {len(active)*0.02:.1f}s")
            print(f"Avg torque : {np.mean(np.abs(active)):.4f}Nm")
    print(f"Log        : {LOG_FILE}")
    print("="*45)

def run_walking_assist(stop_event):
    """
    Your walking script, exactly as given, wrapped in a function so the
    dashboard's HIGH button can start/stop it instead of you launching it
    with `python3 walking.py` / Ctrl+C. Changes vs. your file:
      - `while True` -> `while not stop_event.is_set()`
      - `global` declarations added (required once this code lives inside
        a function — without them check_watchdog()/log_data() wouldn't see
        the live values; this was implicit before since it ran at the top
        level of its own file)
      - state reset at the top so every press starts clean, like relaunching
        the script fresh
      - one telemetry update call per loop + simple rep counter, purely so
        the dashboard has numbers to show
    No threshold, no torque formula, no safety check, no detection logic
    was changed.
    """
    global imu, odrv, axis, LOG_FILE, last_loop_time, cf_angle, last_imu
    global current_direction, ext_candidate_since, motor_engaged
    global session_torques, start_time, watchdog_fault
    global rep_count_walking, rep_peaked_walking

    # reset state — same as relaunching the script fresh
    cf_angle = 0.0
    last_imu = time.time()
    GZ_BUFFER.clear(); GZ_BUFFER.extend([0.0, 0.0, 0.0])
    PITCH_BUFFER.clear(); PITCH_BUFFER.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    current_direction = 0
    ext_candidate_since = None
    motor_engaged = False
    session_torques = []
    watchdog_fault = False
    rep_count_walking = 0
    rep_peaked_walking = False

    update_telemetry(running=True, status="Connecting to IMU / ODrive...")

    print("Initializing IMU...")
    imu = mpu6050(0x68)
    imu.set_accel_range(mpu6050.ACCEL_RANGE_2G)
    imu.set_gyro_range(mpu6050.GYRO_RANGE_250DEG)
    print("IMU ready")

    print("Connecting ODrive...")
    odrv = odrive.find_any()
    axis = odrv.axis0

    axis.motor.config.current_lim    = 3.0
    axis.motor.config.torque_lim     = 0.60
    axis.controller.config.vel_limit = VELOCITY_LIM
    axis.requested_state             = AXIS_STATE_IDLE
    axis.controller.input_torque     = 0.0
    print(f"{MODE_NAME} READY | Vbus: {odrv.vbus_voltage:.1f}V\n")

    LOG_FILE = f"walking_{int(time.time())}.csv"
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow([
            'time', 'pitch', 'gz', 'gz_avg',
            'pitch_trend', 'direction', 'torque', 'status'
        ])

    try:
        start_time = time.time()   # <-- added: so `finally` has a sane value even if stopped during warmup

        print("Warming up IMU — hold still 5 seconds...")
        update_telemetry(status="Warming up IMU — hold still 5 seconds")
        for _ in range(100):
            if stop_event.is_set():        # <-- added: bail out if mode is left mid-warmup
                return
            update_imu()
            time.sleep(0.05)
        print("IMU ready\n")

        print(f"Running {MODE_NAME}")
        print(f"Pitch safety      : {PITCH_SAFE_MAX}° to {PITCH_SAFE_MIN}°")
        print(f"Flex min pitch    : {FLEX_MIN_PITCH}° (no flex assist above this)")
        print(f"Ext min pitch     : {EXT_MIN_PITCH}° (no ext assist above this)")
        print(f"Max flex torque   : {TORQUE_FLEX_MAX} Nm")
        print(f"Max ext torque    : {TORQUE_EXT_MAX} Nm")
        print(f"Gyro threshold    : {GYRO_THRESHOLD} deg/s")
        print(f"Ext confirm time  : {EXT_CONFIRM_TIME}s")
        print(f"Startup lockout   : {STARTUP_LOCKOUT}s\n")
        print("Walk normally.\n")

        start_time = time.time()

        while not stop_event.is_set():          # <-- was: while True
            last_loop_time = time.time()
            check_watchdog()

            t          = time.time() - start_time
            locked_out = t < STARTUP_LOCKOUT

            pitch, gz, gz_avg, pitch_trend = update_imu()
            direction = get_direction(gz_avg, pitch, pitch_trend)
            torque, should_engage = calculate_torque(
                direction, gz, pitch, locked_out
            )

            if should_engage:
                engage_motor()
                axis.controller.input_torque = torque
            else:
                release_motor()

            if locked_out:
                status = f'LOCKOUT {STARTUP_LOCKOUT - t:.1f}s'
            elif pitch < PITCH_SAFE_MAX or pitch > PITCH_SAFE_MIN:
                status = '⚠ PITCH LIMIT'
            elif direction == -1:
                status = 'ASSISTING FLEX'
            elif direction == 1:
                status = 'ASSISTING EXT'
            else:
                status = 'IDLE'

            log_data(t, pitch, gz, gz_avg, pitch_trend,
                     direction, torque, status)
            session_torques.append(torque)

            print(
                f"Pitch:{pitch:7.2f}° trend:{pitch_trend:+5.2f} | "
                f"Gz:{gz:6.1f} avg:{gz_avg:6.1f} | "
                f"Dir:{direction:2d} | "
                f"Torque:{torque:+.3f}Nm | "
                f"{status}"
            )

            # --- added: dashboard rep counter + telemetry push, no effect on control above ---
            angle = max(0.0, -pitch)
            if angle >= DASH_REP_PEAK_ANGLE and not rep_peaked_walking:
                rep_peaked_walking = True
            elif angle < DASH_REP_RESET_ANGLE and rep_peaked_walking:
                rep_peaked_walking = False
                rep_count_walking += 1

            update_telemetry(angle=round(angle, 2), direction=direction,
                              torque=round(torque, 4), status=status,
                              reps=rep_count_walking, running=True)
            # --- end added block ---

            time.sleep(0.02)

    finally:
        axis.controller.input_torque = 0.0
        axis.requested_state         = AXIS_STATE_IDLE
        duration = time.time() - start_time
        print_summary(duration)
        print("SAFE SHUTDOWN ✓")
        final_status = ("WATCHDOG FAULT — press HIGH again to restart"
                         if watchdog_fault else
                         "Stopped — select HIGH to run again")
        update_telemetry(running=False, status=final_status,
                          direction=0, torque=0.0)

# ════════════════════════════════════════════════════════════════════════
# START/STOP WIRING — this is what connects the dashboard's HIGH button to
# the function above. Nothing here touches the control logic itself.
# ════════════════════════════════════════════════════════════════════════
walking_thread = None
walking_stop_event = threading.Event()
walking_control_lock = threading.Lock()

current_mode = 2          # matches the dashboard's default (Medium)
mode_lock = threading.Lock()

def start_walking_code():
    global walking_thread
    with walking_control_lock:
        if walking_thread is not None and walking_thread.is_alive():
            return
        if not HARDWARE_AVAILABLE:
            print("[!] HIGH selected but odrive/mpu6050 aren't installed here — can't run.")
            update_telemetry(status="NO HARDWARE — install odrive + mpu6050 on this machine",
                              running=False)
            return
        walking_stop_event.clear()
        walking_thread = threading.Thread(target=run_walking_assist,
                                           args=(walking_stop_event,), daemon=True)
        walking_thread.start()
        print("[>] HIGH mode selected on dashboard — walking-assist code started.")

def stop_walking_code():
    global walking_thread
    with walking_control_lock:
        if walking_thread is not None and walking_thread.is_alive():
            walking_stop_event.set()
            print("[x] Left HIGH mode — stopping walking-assist code.")
        walking_thread = None

def set_mode(new_mode: int):
    global current_mode
    with mode_lock:
        previous = current_mode
        current_mode = new_mode
    print(f"  Mode changed -> {MODE_NAMES.get(new_mode, new_mode)} (from dashboard)")

    if new_mode == WALKING_CODE_MODE_ID and previous != WALKING_CODE_MODE_ID:
        start_walking_code()
    elif previous == WALKING_CODE_MODE_ID and new_mode != WALKING_CODE_MODE_ID:
        stop_walking_code()

def get_mode() -> int:
    with mode_lock:
        return current_mode

# ─── PLACEHOLDER SENSORS (deliberately flat — see chat reply) ───────────
# No real EMG (ADS1115/MyoWare) or battery-voltage reading code has been
# wired in yet, so these report flat/neutral values instead of fake motion.
# That way nothing moves on the dashboard unless it's coming from your
# actual hardware via run_walking_assist() above.
def read_emg_quad() -> float:
    return 0.0

def read_emg_ham() -> float:
    return 0.0

def read_battery() -> float:
    return 1.0

# ─── WEBSOCKET HANDLER ────────────────────────────────────────────────────
connected_clients: set = set()

async def handle_client(websocket, path=None):
    connected_clients.add(websocket)
    client = websocket.remote_address[0] if websocket.remote_address else "?"
    print(f"\n[+] Web app connected: {client}  (total clients: {len(connected_clients)})")

    async def send_loop():
        interval = 1.0 / SEND_HZ
        while True:
            t0 = time.time()
            tel = read_telemetry()
            max_torque = max(TORQUE_FLEX_MAX, TORQUE_EXT_MAX)
            assist = min(1.0, abs(tel["torque"]) / max_torque) if tel["running"] else 0.0
            payload = {
                "angle":     tel["angle"],
                "emg_quad":  read_emg_quad(),
                "emg_ham":   read_emg_ham(),
                "mode":      get_mode(),
                "reps":      tel["reps"],
                "battery":   read_battery(),
                "assist":    round(assist, 3),
                "status":    tel["status"],
                "timestamp": round(time.time(), 3),
            }
            await websocket.send(json.dumps(payload))
            elapsed = time.time() - t0
            await asyncio.sleep(max(0, interval - elapsed))

    async def recv_loop():
        async for raw_msg in websocket:
            try:
                cmd = json.loads(raw_msg)
                command = cmd.get("command")
                if command == "set_mode":
                    new_mode = int(cmd["value"])
                    if 0 <= new_mode <= 5:
                        set_mode(new_mode)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"  Bad command received: {e}")

    try:
        await asyncio.gather(send_loop(), recv_loop())
    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Web app disconnected: {client}")
    finally:
        connected_clients.discard(websocket)

# ─── HTTP SERVER ──────────────────────────────────────────────────────────
def start_http_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("0.0.0.0", 8080), handler)
    print("  Web UI  : http://0.0.0.0:8080")
    httpd.serve_forever()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 64)
    print("  RAF Exoskeleton — WebSocket Server  (v3 — HIGH = real walking code)")
    print("=" * 64)
    print(f"  Host : {HOST}:{PORT}")
    print(f"  Rate : {SEND_HZ} Hz")
    print(f"  HIGH mode runs your walking script {'(hardware found)' if HARDWARE_AVAILABLE else '— NO HARDWARE FOUND'}")
    print()
    print("  1. Find your Pi IP:   hostname -I")
    print("  2. Open in ANY browser:")
    print("     http://<Pi-IP>:8080/RAF_Monitor_v2.html")
    print("  3. Tap HIGH on the dashboard -> your walking code starts on the real motor.")
    print("     Tap any other mode -> motor releases, walking code stops.")
    print("=" * 64)

    threading.Thread(target=start_http_server, daemon=True).start()

    async with websockets.serve(handle_client, HOST, PORT, ping_interval=None):
        print(f"\n  Listening on ws://0.0.0.0:{PORT} ...\n")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Server stopped.") 
