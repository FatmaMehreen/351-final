import time
import math
import numpy as np
import odrive
from odrive.enums import *
from mpu6050 import mpu6050
import csv

# ═══════════════════════════════════════
# SETTINGS — FROM YOUR CALIBRATION
# ═══════════════════════════════════════
MODE_NAME = "POSITION MATCHING"

STRAIGHT_PITCH = -5.57
STRAIGHT_TURNS = 1.7090
FLEXED_PITCH   = -59.69
FLEXED_TURNS   = -7.3434

TURNS_PER_DEG = (FLEXED_TURNS - STRAIGHT_TURNS) / (FLEXED_PITCH - STRAIGHT_PITCH)
# = -9.0524 / -54.12 = 0.16725

PITCH_SAFE_MIN = -8.0     # near full extension (slightly inside true limit)
PITCH_SAFE_MAX = -57.0    # near full flexion   (slightly inside true limit)

CF_ALPHA        = 0.98
STARTUP_LOCKOUT = 5.0

# Position control tuning — start conservative, increase gradually
VEL_LIMIT_TURNS    = 4.0   # turns/sec max motor speed
ACCEL_LIMIT_TURNS  = 6.0   # turns/sec^2 — how quickly it can ramp speed
POS_FILTER_BW      = 6.0   # Hz — position filter bandwidth (smoothness)

CURRENT_LIM = 6.0
TORQUE_LIM  = 1.5    # generous — position loop needs headroom

# ═══════════════════════════════════════
# IMU INIT
# ═══════════════════════════════════════
print("Initializing IMU...")
imu = mpu6050(0x68)
imu.set_accel_range(mpu6050.ACCEL_RANGE_2G)
imu.set_gyro_range(mpu6050.GYRO_RANGE_250DEG)
print("IMU ready")

# ═══════════════════════════════════════
# ODRIVE INIT — POSITION CONTROL MODE
# ═══════════════════════════════════════
print("Connecting ODrive...")
odrv = odrive.find_any()
axis = odrv.axis0

axis.motor.config.current_lim = CURRENT_LIM
axis.motor.config.torque_lim  = TORQUE_LIM

axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
axis.controller.config.input_mode   = INPUT_MODE_POS_FILTER
axis.controller.config.input_filter_bandwidth = POS_FILTER_BW

axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

# Start target at current actual position — avoids a sudden jump
start_turns = axis.encoder.pos_estimate
axis.controller.input_pos = start_turns

print(f"{MODE_NAME} READY | Vbus: {odrv.vbus_voltage:.1f}V")
print(f"Starting motor position: {start_turns:.4f} turns\n")

# ═══════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════
LOG_FILE = f"position_match_{int(time.time())}.csv"
with open(LOG_FILE, 'w', newline='') as f:
    csv.writer(f).writerow([
        'time', 'pitch', 'target_turns', 'actual_turns', 'status'
    ])

def log_data(t, pitch, target_turns, actual_turns, status):
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            round(t, 3), round(pitch, 2),
            round(target_turns, 4), round(actual_turns, 4),
            status
        ])

# ═══════════════════════════════════════
# WATCHDOG
# ═══════════════════════════════════════
last_loop_time = time.time()

def check_watchdog():
    if time.time() - last_loop_time > 1.0:
        axis.requested_state = AXIS_STATE_IDLE
        print("WATCHDOG TRIGGERED — motor cut")
        exit()

# ═══════════════════════════════════════
# IMU — COMPLEMENTARY FILTER
# ═══════════════════════════════════════
cf_angle  = 0.0
last_time = time.time()

def get_accel_angle(accel):
    ax, ay, az = accel['x'], accel['y'], accel['z']
    return math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))

def update_imu():
    global cf_angle, last_time
    now       = time.time()
    dt        = min(now - last_time, 0.1)
    last_time = now

    accel = imu.get_accel_data()
    gyro  = imu.get_gyro_data()
    gz    = -gyro['z']

    accel_angle = get_accel_angle(accel)
    cf_angle = (CF_ALPHA * (cf_angle + gz * dt) +
                (1 - CF_ALPHA) * accel_angle)

    return cf_angle, gz

# ═══════════════════════════════════════
# PITCH → TARGET MOTOR TURNS MAPPING
# ═══════════════════════════════════════
def pitch_to_turns(pitch):
    """Maps current leg pitch to corresponding motor turns target"""
    return STRAIGHT_TURNS + (pitch - STRAIGHT_PITCH) * TURNS_PER_DEG

# ═══════════════════════════════════════
# SAFETY CLAMP ON TARGET POSITION
# ═══════════════════════════════════════
def clamp_target(target_turns):
    """
    Clamp target turns to correspond with safe pitch range —
    prevents commanding motor beyond safe leg range even
    if IMU briefly reads a bad/noisy value
    """
    safe_min_turns = pitch_to_turns(PITCH_SAFE_MIN)
    safe_max_turns = pitch_to_turns(PITCH_SAFE_MAX)

    lo, hi = min(safe_min_turns, safe_max_turns), max(safe_min_turns, safe_max_turns)
    return max(lo, min(hi, target_turns))

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
print("Warming up IMU — hold still 5 seconds...")
for _ in range(100):
    update_imu()
    time.sleep(0.05)
print("IMU ready\n")

print(f"Running {MODE_NAME}")
print(f"TURNS_PER_DEG  : {TURNS_PER_DEG:.5f}")
print(f"Pitch safe range: {PITCH_SAFE_MIN}° to {PITCH_SAFE_MAX}°")
print(f"Velocity limit  : {VEL_LIMIT_TURNS} turns/s")
print(f"Startup lockout : {STARTUP_LOCKOUT}s\n")

start_time = time.time()

try:
    while True:
        last_loop_time = time.time()
        check_watchdog()

        t          = time.time() - start_time
        locked_out = t < STARTUP_LOCKOUT

        pitch, gz = update_imu()

        if locked_out:
            # Hold at current actual position during lockout
            target_turns = axis.encoder.pos_estimate
            status = f'LOCKOUT {STARTUP_LOCKOUT - t:.1f}s'
        else:
            raw_target   = pitch_to_turns(pitch)
            target_turns = clamp_target(raw_target)

            if pitch < PITCH_SAFE_MAX or pitch > PITCH_SAFE_MIN:
                status = '⚠ NEAR PITCH LIMIT — clamped'
            else:
                status = 'TRACKING'

        axis.controller.input_pos = target_turns
        actual_turns = axis.encoder.pos_estimate

        log_data(t, pitch, target_turns, actual_turns, status)

        print(
            f"Pitch:{pitch:7.2f}° | "
            f"Target:{target_turns:8.4f} | "
            f"Actual:{actual_turns:8.4f} | "
            f"Err:{target_turns - actual_turns:7.4f} | "
            f"{status}"
        )

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    axis.requested_state = AXIS_STATE_IDLE
    print("SAFE SHUTDOWN")

