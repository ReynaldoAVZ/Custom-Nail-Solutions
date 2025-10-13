# signal pin 32
# ls /sys/class/pwm/pwmchip1/ -1
# Run in terminal: sudo python3 /home/orangepi/Documents/repos/Custom-Nail-Solutions/Test\ Scripts/servo_pwm.py
# --- Configuration ---

import os
import time

# PWM configuration for 500 Hz digital servo
PWMCHIP = "/sys/class/pwm/pwmchip1"
PWM = "pwm0"
PERIOD_NS = 2_000_000    # 2 ms period = 500 Hz
MIN_PULSE = 120_000    # Left most angle (8.1 deg)
MAX_PULSE = 1_060_000    # Right most angle (must be < period) (92.7 deg) (1030/2000)*180 = 92.7
STEP_NS = 4_000           # 4 µs per step (smooth motion)
STEP_DELAY = 0.0007         # 4 ms between update (controls sweep speed) max = 0.0007

def write(path, value):
    with open(path, "w") as f:
        f.write(str(int(value)))

def set_pulse(ns):
    write(f"{PWMCHIP}/{PWM}/duty_cycle", ns)

def ensure_exported():
    if not os.path.exists(f"{PWMCHIP}/{PWM}"):
        write(f"{PWMCHIP}/export", "0")
        time.sleep(0.5)

# Validate pulse range
if MAX_PULSE >= PERIOD_NS:
    raise ValueError(f"MAX_PULSE ({MAX_PULSE}) must be less than PERIOD_NS ({PERIOD_NS}).")

try:
    ensure_exported()

    try:
        write(f"{PWMCHIP}/{PWM}/enable", "0")
    except Exception:
        pass

    write(f"{PWMCHIP}/{PWM}/period", PERIOD_NS)
    set_pulse(MIN_PULSE)
    time.sleep(0.05)
    write(f"{PWMCHIP}/{PWM}/enable", "1")

except PermissionError:
    print("Permission denied. Run this script with sudo.")
    exit(1)
except FileNotFoundError:
    print("PWM sysfs interface not found.")
    exit(1)

print("Performing single slow sweep at 500 Hz...")

try:
    # Sweep forward
    for pulse in range(MIN_PULSE, MAX_PULSE + 1, STEP_NS):
        set_pulse(pulse)
        time.sleep(STEP_DELAY)

    time.sleep(0.75)

    # Sweep back
    for pulse in range(MAX_PULSE, MIN_PULSE - 1, -STEP_NS):
        set_pulse(pulse)
        time.sleep(STEP_DELAY)

    print("Sweep complete.")

finally:
    # Safely disable PWM
    try:
        write(f"{PWMCHIP}/{PWM}/enable", "0")
        print("PWM disabled safely.")
    except Exception as e:
        print(f"Error disabling PWM: {e}")

