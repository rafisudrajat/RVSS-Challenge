import cv2
import numpy as np

def calculate_action(Kp, Ki, Kd, d_left, d_right, stop_sign_detected, e_prev=0, integral=0):
    
    # --- CONSTANTS ---
    BASE_SPEED = 15      
    TURN_BRAKING = 0.3    
    MAX_INTEGRAL = 10.0   # Anti-windup limit (Tune this!)

    # --- 1. PRIORITY: STOP SIGN ---
    if stop_sign_detected:
        # Return 0 speed, reset error and integral
        return 0, 0, 0, 0 

    # --- 2. HANDLE MISSING EDGES (90-Degree Turns) ---
    if d_left is None:
        # We are panic turning. Reset integral so we don't build up error while blind.
        return 0.2, 0.6, e_prev, 0 
    if d_right is None:
        return 0.6, 0.2, e_prev, 0

    # --- 3. CALCULATE STEERING ERROR ---
    # Normalized error: Range [-1.0 to 1.0]
    # If error is positive, we are too far to the right, (need to steer left)
    error = (d_left - d_right) / ( abs(d_left + d_right) + 1e-6)
    
    # Proportional
    P = Kp * error

    # Integral (with Anti-Windup)
    integral = integral + error  # Simplified accumulation
    # Clamp the integral to prevent it from growing too large
    integral = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, integral))
    I = Ki * integral

    # Derivative
    D = Kd * (error - e_prev)
    
    steering_adjustment = P + I + D

    # Update state for next loop
    e_prev = error
    
    # --- 4. CALCULATE DYNAMIC SPEED ---
    current_speed = BASE_SPEED - (abs(error) * TURN_BRAKING)
    
    # --- 5. MIX MOTORS ---
    left_motor = current_speed - steering_adjustment
    right_motor = current_speed + steering_adjustment
    
    left_motor = max(0, min(1, left_motor))
    right_motor = max(0, min(1, right_motor))
    
    return left_motor, right_motor, e_prev, integral