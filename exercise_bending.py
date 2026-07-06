BENDONLY.PY

import time
import math
import numpy as np
import odrive
from odrive.enums import *
from mpu6050 import mpu6050
from collections import deque
import csv

# ═══════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════
MODE_NAME = "RAF EXOSKELETON"

PITCH_SAFE_MIN = -5.0    # 7° above your standing pitch of -12°
PITCH_SAFE_MAX = -62.0   # hard cutoff at max flex

TORQUE_FLEX = 0.3        # Nm — assisting knee bend
TORQUE_EXT  = 0.25       # Nm — assisting knee straighten

VELOCITY_LIM    = 6.0
GYRO_THRESHOLD  = 12     # deg/s — must exceed this to assist
GYRO_HYSTERESIS = 4      # prevents flickering at threshold edge
STARTUP_LOCKOUT = 5.0
CF_ALPHA        = 0.98

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
# Start in IDLE — only engage when actually assisting
axis.requested_state             = AXIS_STATE_IDLE
axis.controller.input_torque     = 0.0
print(f"{MODE_NAME} READY | Vbus: {odrv.vbus_voltage:.1f}V\n")

# ═══════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════
LOG_FILE = f"walking_{int(time.time())}.csv"
with open(LOG_FILE, 'w', newline='') as f:
    csv.writer(f).writerow([
        'time', 'pitch', 'gz', 'direction', 'torque', 'status'
    ])

def log_data(t, pitch, gz, direction, torque, status):
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            round(t, 3), round(pitch, 2),
            round(gz, 2), direction,
            round(torque, 4), status
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
# IMU
# gz negative = knee flexing  (confirmed)
# gz positive = knee extending (confirmed)
# ═══════════════════════════════════════
cf_angle   = 0.0
last_time  = time.time()
GZ_BUFFER  = deque(maxlen=3)  # small buffer, keeps response fast

def get_accel_angle(accel):
    ax = accel['x']
    ay = accel['y']
    az = accel['z']
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

last_direction = 0

def get_direction(gz):
    global last_direction
    GZ_BUFFER.append(gz)
    avg = sum(GZ_BUFFER) / len(GZ_BUFFER)

    if avg < -(GYRO_THRESHOLD + GYRO_HYSTERESIS):
        last_direction = -1   # flexion
    elif avg > (GYRO_THRESHOLD + GYRO_HYSTERESIS):
        last_direction = 1    # extension
    elif abs(avg) < (GYRO_THRESHOLD - GYRO_HYSTERESIS):
        last_direction = 0    # still

    return last_direction

# ═══════════════════════════════════════
# MOTOR ENGAGE / RELEASE
# Motor goes IDLE when not assisting —
# this means zero resistance when still
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
# TORQUE — instant, no ramp
# gz negative → flex torque (negative)
# gz positive → ext torque  (positive)
# zero instantly when leg stops
# ═══════════════════════════════════════
def calculate_torque(direction, pitch, locked_out):

    if locked_out:
        return 0.0, False

    # Safety cutoff
    if pitch < PITCH_SAFE_MAX or pitch > PITCH_SAFE_MIN:
        return 0.0, False

    if direction == -1:
        # Knee flexing — apply negative torque (flex direction)
        return -TORQUE_FLEX, True

    elif direction == 1:
        # Knee extending — apply positive torque (ext direction)
        return TORQUE_EXT, True

    else:
        return 0.0, False

# ═══════════════════════════════════════
# SESSION SUMMARY
# ═══════════════════════════════════════
session_torques = []

def print_summary(duration):
    print("\n" + "="*40)
    print(f"  {MODE_NAME} SUMMARY")
    print("="*40)
    print(f"Duration   : {duration:.1f}s")
    if session_torques:
        active = [t for t in session_torques if abs(t) > 0.001]
        if active:
            print(f"Active time: {len(active)*0.02:.1f}s")
            print(f"Avg torque : {np.mean(np.abs(active)):.4f}Nm")
    print(f"Log        : {LOG_FILE}")
    print("="*40)

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
print("Warming up IMU — hold still 5 seconds...")
for _ in range(100):
    update_imu()
    time.sleep(0.05)
print("IMU ready\n")

print(f"Running {MODE_NAME}")
print(f"Pitch safety range : {PITCH_SAFE_MAX}° to {PITCH_SAFE_MIN}°")
print(f"Flexion torque     : {TORQUE_FLEX} Nm")
print(f"Extension torque   : {TORQUE_EXT} Nm")
print(f"Gyro threshold     : {GYRO_THRESHOLD} deg/s")
print(f"Startup lockout    : {STARTUP_LOCKOUT}s\n")

start_time = time.time()

try:
    while True:
        last_loop_time = time.time()
        check_watchdog()

        t          = time.time() - start_time
        locked_out = t < STARTUP_LOCKOUT

        pitch, gz = update_imu()
        direction  = get_direction(gz)

        torque, should_engage = calculate_torque(direction, pitch, locked_out)

        if should_engage:
            engage_motor()
            axis.controller.input_torque = torque
        else:
            release_motor()

        if locked_out:
            status = f'LOCKOUT {STARTUP_LOCKOUT - t:.1f}s'
        elif pitch < PITCH_SAFE_MAX or pitch > PITCH_SAFE_MIN:
            status = '⚠ PITCH LIMIT — motor off'
        elif direction == -1:
            status = 'ASSISTING FLEX'
        elif direction == 1:
            status = 'ASSISTING EXTENSION'
        else:
            status = 'IDLE — no resistance'

        log_data(t, pitch, gz, direction, torque, status)
        session_torques.append(torque)

        print(
            f"Pitch:{pitch:7.2f}° | "
            f"Gz:{gz:6.1f} | "
            f"Dir:{direction:2d} | "
            f"Torque:{torque:+.3f}Nm | "
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

