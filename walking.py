"""
RAF EXOSKELETON — WALKING ASSIST v5
======================================
Fixes from v4:

FIX 1 — Extension only when leg is genuinely extending
  Not when leg decelerates after flex.
  Requires gz > threshold AND pitch_trend positive AND
  pitch is below -15° (leg has actually bent before extending).
  This means: can only get extension assist if you were
  actually bent first. Prevents false extension at standstill.

FIX 2 — No flex assist at heel strike
  At heel strike, pitch is near straight (-5° to -15°).
  Impact sends negative gz spike but pitch is still near straight.
  Rule: flex assist only allowed when pitch < FLEX_MIN_PITCH.
  This means: no flex assist unless knee is already somewhat bent.
  Heel strike impact is ignored.

RESULT:
  Standing still  → IDLE, zero torque, no resistance
  Knee bending    → FLEX assist only after pitch drops past -15°
  Knee extending  → EXT assist only if pitch was bent AND trend rising
  Heel strike     → ignored, motor stays IDLE
"""

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
MODE_NAME = "RAF EXOSKELETON — WALKING v5"

PITCH_SAFE_MIN   = -5.0    # hard safety cutoff — too straight
PITCH_SAFE_MAX   = -62.0   # hard safety cutoff — too bent

# Flex assist only allowed when pitch is below this
# Prevents heel strike impact (near-straight pitch) from
# triggering flex torque
FLEX_MIN_PITCH   = -15.0   # must be at least this bent to get flex assist

# Extension assist only allowed when pitch is below this
# Ensures leg was actually bent before we assist straightening
EXT_MIN_PITCH    = -13.0   # must have been bent at least this much

TORQUE_FLEX_MAX  = 0.3     # Nm max flex assist
TORQUE_EXT_MAX   = 0.25    # Nm max extension assist
TORQUE_MIN       = 0.08    # Nm minimum to actually move motor

GZ_MAX           = 40.0    # deg/s at which max torque is reached

VELOCITY_LIM     = 6.0
GYRO_THRESHOLD   = 10      # deg/s — minimum gz to trigger
GYRO_HYSTERESIS  = 3

# Extension requires gz to be sustained for this long
# Prevents deceleration spike from triggering extension
EXT_CONFIRM_TIME = 0.25    # seconds

STARTUP_LOCKOUT  = 5.0
CF_ALPHA         = 0.98

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
LOG_FILE = f"walking_{int(time.time())}.csv"
with open(LOG_FILE, 'w', newline='') as f:
    csv.writer(f).writerow([
        'time', 'pitch', 'gz', 'gz_avg',
        'pitch_trend', 'direction', 'torque', 'status'
    ])

def log_data(t, pitch, gz, gz_avg, trend, direction, torque, status):
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            round(t, 3), round(pitch, 2), round(gz, 2),
            round(gz_avg, 2), round(trend, 3),
            direction, round(torque, 4), status
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
    cf_angle = (CF_ALPHA * (cf_angle + gz * dt) +
                (1 - CF_ALPHA) * get_accel_angle(accel))

    GZ_BUFFER.append(gz)
    PITCH_BUFFER.append(cf_angle)

    gz_avg      = sum(GZ_BUFFER) / len(GZ_BUFFER)
    pitch_trend = PITCH_BUFFER[-1] - PITCH_BUFFER[0]

    return cf_angle, gz, gz_avg, pitch_trend

# ═══════════════════════════════════════
# DIRECTION DETECTION
#
# FLEX:
#   gz_avg < -threshold
#   AND pitch < FLEX_MIN_PITCH   ← knee must already be bent
#   → triggers immediately
#
# EXTENSION:
#   gz_avg > +threshold
#   AND pitch_trend positive      ← pitch genuinely rising
#   AND pitch < EXT_MIN_PITCH     ← was actually bent before
#   AND sustained for EXT_CONFIRM_TIME ← not just a spike
#   → triggers after confirmation
#
# IDLE:
#   gz drops below threshold
#   → immediately goes idle (no hold — stops instantly)
# ═══════════════════════════════════════
current_direction = 0
ext_candidate_since = None   # when we first saw a valid ext signal

def get_direction(gz_avg, pitch, pitch_trend):
    global current_direction, ext_candidate_since
    now = time.time()

    ON  = GYRO_THRESHOLD + GYRO_HYSTERESIS   # 13 deg/s
    OFF = GYRO_THRESHOLD - GYRO_HYSTERESIS   # 7  deg/s

    # ── FLEX detection ──────────────────────────────────
    # gz negative + knee already bent (not at heel strike)
    flex_signal = (gz_avg < -ON) and (pitch < FLEX_MIN_PITCH)

    # ── EXTENSION detection ─────────────────────────────
    # gz positive + pitch rising + was bent + sustained
    ext_signal = (
        gz_avg > ON and
        pitch_trend > 0.1 and
        pitch < EXT_MIN_PITCH
    )

    if ext_signal:
        if ext_candidate_since is None:
            ext_candidate_since = now
        # Only confirm extension if sustained long enough
        ext_confirmed = (now - ext_candidate_since) >= EXT_CONFIRM_TIME
    else:
        ext_candidate_since = None
        ext_confirmed = False

    # ── Apply direction ──────────────────────────────────
    if flex_signal:
        current_direction = -1
        ext_candidate_since = None   # reset ext candidate on flex

    elif ext_confirmed:
        current_direction = 1

    elif abs(gz_avg) < OFF:
        # gz dropped — go idle immediately
        current_direction = 0
        ext_candidate_since = None

    return current_direction

# ═══════════════════════════════════════
# MOTOR ENGAGE / RELEASE
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
# TORQUE — scales with gz speed
# ═══════════════════════════════════════
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

# ═══════════════════════════════════════
# SESSION SUMMARY
# ═══════════════════════════════════════
session_torques = []

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

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
print("Warming up IMU — hold still 5 seconds...")
for _ in range(100):
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

try:
    while True:
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

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    axis.controller.input_torque = 0.0
    axis.requested_state         = AXIS_STATE_IDLE
    duration = time.time() - start_time
    print_summary(duration)
    print("SAFE SHUTDOWN ✓")

