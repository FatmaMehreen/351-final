import time
import math
import numpy as np
import odrive
from odrive.enums import *
from mpu6050 import mpu6050
from collections import deque
import csv
import tflite_runtime.interpreter as tflite

# ═══════════════════════════════════════
# SETTINGS — same safety values as v5
# ═══════════════════════════════════════
MODE_NAME = "RAF EXOSKELETON — v5 + ML PHASE MODIFIER"

PITCH_SAFE_MIN   = -5.0
PITCH_SAFE_MAX   = -62.0
FLEX_MIN_PITCH   = -15.0
EXT_MIN_PITCH    = -13.0

TORQUE_FLEX_MAX  = 0.3
TORQUE_EXT_MAX   = 0.25
TORQUE_MIN       = 0.08
GZ_MAX           = 40.0

VELOCITY_LIM     = 6.0
GYRO_THRESHOLD   = 10
GYRO_HYSTERESIS  = 3
EXT_CONFIRM_TIME = 0.25
STARTUP_LOCKOUT  = 5.0
CF_ALPHA         = 0.98

# ── ML phase confidence gate ──
# Below this, ML modifier is ignored — v5 logic runs unmodified
CONFIDENCE_MIN = 0.55

# ── ML phase → torque scale factor ──
# These MULTIPLY the v5 torque, they don't replace it
# 1.0 = no change from v5 behavior
PHASE_SCALE = {
    'INITIAL_SWING':  1.15,   # slightly boost during confirmed swing
    'TERMINAL_SWING': 1.10,
    'PRE_SWING':       0.90,  # slightly ease — transition, be gentle
    'STANCE':          0.70,  # reduce torque — shouldn't be assisting much in stance
    'STILL':           0.0,   # zero out — no assist when truly still
}

WINDOW   = 20
FEATURES_EXPECTED = list(np.load('feature_list.npy', allow_pickle=True))
print(f"Model expects features in this order: {FEATURES_EXPECTED}")

# ═══════════════════════════════════════
# LOAD ML MODEL
# ═══════════════════════════════════════
print("Loading ML model...")
interpreter  = tflite.Interpreter(model_path='gait_model.tflite')
interpreter.allocate_tensors()
input_det    = interpreter.get_input_details()
output_det   = interpreter.get_output_details()
SCALER_MEAN  = np.load('scaler_mean.npy')
SCALER_SCALE = np.load('scaler_scale.npy')
CLASSES      = np.load('label_classes.npy', allow_pickle=True)
print(f"Model loaded | Classes: {list(CLASSES)}")
print(f"Input shape expected: {input_det[0]['shape']}")

def predict_phase(window_array):
    """window_array shape must be (WINDOW, num_features) in FEATURES_EXPECTED order"""
    normed = (window_array - SCALER_MEAN) / SCALER_SCALE
    inp    = normed.astype(np.float32)[np.newaxis, ...]
    interpreter.set_tensor(input_det[0]['index'], inp)
    interpreter.invoke()
    probs  = interpreter.get_tensor(output_det[0]['index'])[0]
    idx    = np.argmax(probs)
    return CLASSES[idx], float(probs[idx])

# ═══════════════════════════════════════
# IMU INIT
# ═══════════════════════════════════════
print("Initializing IMU...")
imu = mpu6050(0x68)
imu.set_accel_range(mpu6050.ACCEL_RANGE_2G)
imu.set_gyro_range(mpu6050.GYRO_RANGE_250DEG)
print("IMU ready")

# ═══════════════════════════════════════
# ODRIVE INIT
# ═══════════════════════════════════════
print("Connecting ODrive...")
odrv = odrive.find_any()
axis = odrv.axis0

axis.motor.config.current_lim    = 3.0
axis.motor.config.torque_lim     = 0.35
axis.controller.config.vel_limit = VELOCITY_LIM
axis.requested_state             = AXIS_STATE_IDLE
axis.controller.input_torque     = 0.0
print(f"{MODE_NAME} READY | Vbus: {odrv.vbus_voltage:.1f}V\n")

# ═══════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════
LOG_FILE = f"hybrid_session_{int(time.time())}.csv"
with open(LOG_FILE, 'w', newline='') as f:
    csv.writer(f).writerow([
        'time', 'pitch', 'gz', 'gz_avg', 'pitch_trend',
        'direction', 'ml_phase', 'ml_conf',
        'base_torque', 'final_torque', 'status'
    ])

def log_data(t, pitch, gz, gz_avg, trend, direction,
             ml_phase, ml_conf, base_torque, final_torque, status):
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            round(t, 3), round(pitch, 2), round(gz, 2),
            round(gz_avg, 2), round(trend, 3), direction,
            ml_phase, round(ml_conf, 3),
            round(base_torque, 4), round(final_torque, 4), status
        ])

# ═══════════════════════════════════════
# WATCHDOG
# ═══════════════════════════════════════
last_loop_time = time.time()

def check_watchdog():
    if time.time() - last_loop_time > 1.0:
        axis.controller.input_torque = 0.0
        axis.requested_state         = AXIS_STATE_IDLE
        print("WATCHDOG TRIGGERED — motor cut")
        exit()

# ═══════════════════════════════════════
# IMU — same as v5, but also returns raw
# gx, gy, ax, ay, az for ML features
# ═══════════════════════════════════════
cf_angle     = 0.0
last_imu     = time.time()
GZ_BUFFER    = deque([0.0, 0.0, 0.0], maxlen=3)
PITCH_BUFFER = deque([0.0, 0.0, 0.0, 0.0, 0.0], maxlen=5)

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
    gx       =  gyro['x']
    gy       =  gyro['y']
    ax       =  accel['x']
    ay       =  accel['y']
    az       =  accel['z']
    cf_angle = (CF_ALPHA * (cf_angle + gz * dt) +
                (1 - CF_ALPHA) * get_accel_angle(accel))

    GZ_BUFFER.append(gz)
    PITCH_BUFFER.append(cf_angle)

    gz_avg      = sum(GZ_BUFFER) / len(GZ_BUFFER)
    pitch_trend = PITCH_BUFFER[-1] - PITCH_BUFFER[0]

    return cf_angle, gz, gx, gy, ax, ay, az, gz_avg, pitch_trend

# ═══════════════════════════════════════
# FEATURE VECTOR BUILDER — must match
# FEATURES_EXPECTED order from training
# ═══════════════════════════════════════
def build_feature_row(pitch, gz, gx, gy, ax, ay, az, gz_avg, pitch_trend):
    """Returns dict, then we'll select in correct order"""
    return {
        'pitch': pitch, 'gz': gz, 'gx': gx, 'gy': gy,
        'ax': ax, 'ay': ay, 'az': az,
        'gz_avg': gz_avg, 'pitch_trend': pitch_trend
    }

feature_window = deque(maxlen=WINDOW)

# ═══════════════════════════════════════
# v5 DIRECTION DETECTION — UNCHANGED LOGIC
# This remains the PRIMARY safety gate
# ═══════════════════════════════════════
current_direction    = 0
ext_candidate_since  = None

def get_direction(gz_avg, pitch, pitch_trend):
    global current_direction, ext_candidate_since
    now = time.time()

    ON  = GYRO_THRESHOLD + GYRO_HYSTERESIS
    OFF = GYRO_THRESHOLD - GYRO_HYSTERESIS

    flex_signal = (gz_avg < -ON) and (pitch < FLEX_MIN_PITCH)
    ext_signal  = (gz_avg > ON and pitch_trend > 0.1 and pitch < EXT_MIN_PITCH)

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

# ═══════════════════════════════════════
# MOTOR ENGAGE / RELEASE — unchanged
# ═══════════════════════════════════════
motor_engaged = False

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

# ═══════════════════════════════════════
# TORQUE — v5 base calculation, THEN
# ML phase scaling applied on top
# ═══════════════════════════════════════
def calculate_base_torque(direction, gz, pitch, locked_out):
    """This is identical to v5 — the safety-validated logic"""
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

def apply_ml_modifier(base_torque, should_engage, ml_phase, ml_conf):
    """
    ML only SCALES the already-safe v5 torque.
    It can never introduce torque in a direction v5 didn't
    already decide was safe, and it's ignored below confidence
    threshold — v5 behavior is unaffected in that case.
    """
    if not should_engage or base_torque == 0.0:
        return base_torque  # nothing to scale

    if ml_conf < CONFIDENCE_MIN:
        return base_torque  # not confident enough — trust v5 as-is

    scale_factor = PHASE_SCALE.get(ml_phase, 1.0)
    return base_torque * scale_factor

# ═══════════════════════════════════════
# SESSION SUMMARY
# ═══════════════════════════════════════
session_torques = []
session_phases  = []

def print_summary(duration):
    print("\n" + "="*50)
    print(f"  {MODE_NAME} SUMMARY")
    print("="*50)
    print(f"Duration   : {duration:.1f}s")
    if session_torques:
        active = [t for t in session_torques if abs(t) > 0.001]
        if active:
            print(f"Active time: {len(active)*0.02:.1f}s")
            print(f"Avg torque : {np.mean(np.abs(active)):.4f}Nm")
    if session_phases:
        from collections import Counter
        counts = Counter(session_phases)
        print("ML phase distribution during active assist:")
        for ph, c in counts.items():
            print(f"  {ph:18s}: {c}")
    print(f"Log        : {LOG_FILE}")
    print("="*50)

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
print("Warming up IMU — hold still 5 seconds...")
for _ in range(100):
    pitch, gz, gx, gy, ax, ay, az, gz_avg, pitch_trend = update_imu()
    row = build_feature_row(pitch, gz, gx, gy, ax, ay, az, gz_avg, pitch_trend)
    feature_window.append([row[f] for f in FEATURES_EXPECTED])
    time.sleep(0.05)
print("IMU ready\n")

print(f"Running {MODE_NAME}")
print(f"Pitch safety      : {PITCH_SAFE_MAX}° to {PITCH_SAFE_MIN}°")
print(f"ML confidence min : {CONFIDENCE_MIN:.0%}")
print(f"Startup lockout   : {STARTUP_LOCKOUT}s")
print("\nWalk normally.\n")

start_time  = time.time()
ml_phase    = 'STILL'
ml_conf     = 0.0

try:
    while True:
        last_loop_time = time.time()
        check_watchdog()

        t          = time.time() - start_time
        locked_out = t < STARTUP_LOCKOUT

        pitch, gz, gx, gy, ax, ay, az, gz_avg, pitch_trend = update_imu()

        row = build_feature_row(pitch, gz, gx, gy, ax, ay, az, gz_avg, pitch_trend)
        feature_window.append([row[f] for f in FEATURES_EXPECTED])

        if len(feature_window) == WINDOW:
            ml_phase, ml_conf = predict_phase(np.array(feature_window))

        # ── v5 direction + base torque — primary safety logic ──
        direction = get_direction(gz_avg, pitch, pitch_trend)
        base_torque, should_engage = calculate_base_torque(
            direction, gz, pitch, locked_out
        )

        # ── ML modifier — secondary scaling only ──
        final_torque = apply_ml_modifier(
            base_torque, should_engage, ml_phase, ml_conf
        )

        if should_engage and final_torque != 0.0:
            engage_motor()
            axis.controller.input_torque = final_torque
        else:
            release_motor()

        if locked_out:
            status = f'LOCKOUT {STARTUP_LOCKOUT - t:.1f}s'
        elif pitch < PITCH_SAFE_MAX or pitch > PITCH_SAFE_MIN:
            status = '⚠ PITCH LIMIT'
        elif direction == -1:
            status = f'FLEX (ML:{ml_phase})'
        elif direction == 1:
            status = f'EXT (ML:{ml_phase})'
        else:
            status = 'IDLE'

        log_data(t, pitch, gz, gz_avg, pitch_trend, direction,
                 ml_phase, ml_conf, base_torque, final_torque, status)
        session_torques.append(final_torque)
        if should_engage:
            session_phases.append(ml_phase)

        print(
            f"Pitch:{pitch:7.2f}° | "
            f"Dir:{direction:2d} | "
            f"ML:{ml_phase:16s}({ml_conf:.0%}) | "
            f"Base:{base_torque:+.3f} → Final:{final_torque:+.3f}Nm | "
            f"{status}"
        )

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    axis.controller.input_torque = 0.0
    axis.requested_state         = AXIS_STATE_IDLE
    duration = time.time() - start_time
    print_summary(duration)
    print("SAFE SHUTDOWN ✓")


