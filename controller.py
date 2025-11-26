import numpy as np
from numpy.typing import ArrayLike
from racetrack import RaceTrack

class PIDController:
    """Simple PID controller."""
    def __init__(self, Kp, Ki, Kd, windup_limit=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.windup_limit = windup_limit
        self.integral = 0.0
        self.prev_error = None
        
    def compute(self, error, dt=0.05):
        # Proportional
        P = self.Kp * error
        
        # Integral
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        I = self.Ki * self.integral
        
        # Derivative
        if self.prev_error is None:
            D = 0.0
        else:
            D = self.Kd * (error - self.prev_error) / dt
        
        self.prev_error = error
        
        return P + I + D
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = None


# Steering: PD (no integral)
steering_pid = PIDController(Kp=1.5, Ki=0.0, Kd=0.25)
# Velocity: pure P (fast tracking to desired speed)
velocity_pid = PIDController(Kp=3.0, Ki=0.0, Kd=0.0)


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level controller that tracks desired steering angle and velocity.
    
    Args:
        state:    [x, y, steering_angle, velocity, heading]
        desired:  [desired_steering_angle, desired_velocity, ...]
        parameters: vehicle parameters
    
    Returns:
        [steering_velocity, acceleration]
    """
    assert desired.shape[0] >= 2  # At least steering and velocity
    
    current_steering = float(state[2])
    current_velocity = float(state[3])
    
    desired_steering = float(desired[0])
    desired_velocity = float(desired[1])
    
    # Angle-aware steering error in [-pi, pi]
    steering_error = desired_steering - current_steering
    steering_error = np.arctan2(np.sin(steering_error), np.cos(steering_error))
    
    velocity_error = desired_velocity - current_velocity
    
    dt_ctrl = 0.05
    
    steering_velocity = steering_pid.compute(steering_error, dt=dt_ctrl)
    acceleration      = velocity_pid.compute(velocity_error, dt=dt_ctrl)
    
    max_steering_vel = float(parameters[9])
    max_acceleration = float(parameters[10])
    
    steering_velocity = np.clip(steering_velocity, -max_steering_vel, max_steering_vel)
    acceleration      = np.clip(acceleration,      -max_acceleration, max_acceleration)
    
    return np.array([steering_velocity, acceleration])


def compute_curvature_lookahead(centerline, start_idx, num_points):
    """
    Approximate average curvature over the next `num_points` samples
    starting from `start_idx`.

    Curvature ≈ total heading change / total arc length  [rad/m].
    """
    num_centerline = len(centerline)
    if num_centerline < 3 or num_points < 3:
        return 0.0

    window = min(num_points, num_centerline)

    headings = []
    arc_length = 0.0

    for i in range(window - 1):
        idx1 = (start_idx + i) % num_centerline
        idx2 = (start_idx + i + 1) % num_centerline

        p1 = centerline[idx1]
        p2 = centerline[idx2]

        v = p2 - p1
        seg_len = np.linalg.norm(v)

        if seg_len > 1e-6:
            arc_length += seg_len
            headings.append(np.arctan2(v[1], v[0]))

    if arc_length < 1e-6 or len(headings) < 2:
        return 0.0

    total_heading_change = 0.0
    for i in range(len(headings) - 1):
        dtheta = headings[i+1] - headings[i]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        total_heading_change += abs(dtheta)

    return total_heading_change / arc_length


def find_index_at_distance(centerline, start_idx, distance):
    """
    Walk along the centerline from start_idx until `distance` is reached.
    Return the index at that point.
    """
    num_centerline = len(centerline)
    idx = start_idx
    acc = 0.0

    while acc < distance:
        next_idx = (idx + 1) % num_centerline
        seg_len = np.linalg.norm(centerline[next_idx] - centerline[idx])
        acc += seg_len
        idx = next_idx
        if idx == start_idx and acc > 0.0:
            break

    return idx

smooth_straightness = 0.0
curve_exit_hold = 0
tight_turn_hold = 0

def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller that generates desired steering angle and velocity.
    Steering: assignment-based linearized formula.
    Velocity: curvature-based profile + boundary safety, with forward lookahead.
    """
    global smooth_straightness
    global curve_exit_hold
    global tight_turn_hold
    # Extract current state
    x, y = float(state[0]), float(state[1])
    current_velocity = float(state[3])
    heading = float(state[4])           # φ[i]
    wheelbase = float(parameters[0])    # l_wb
    
    car_position = np.array([x, y])
    
    # --- Step 1: closest point on centerline ---
    centerline = racetrack.centerline
    distances = np.linalg.norm(centerline - car_position, axis=1)
    closest_idx = int(np.argmin(distances))
    num_points = len(centerline)

    # --- Step 2: local curvature just to classify straight vs curve ---
    local_curvature = compute_curvature_lookahead(centerline, closest_idx, 12)
    curvature_threshold = 0.01  # radians per meter
    is_straight = local_curvature < curvature_threshold

    max_velocity  = float(parameters[5])
    v_min = float(parameters[2])
    v_straight = max_velocity

    # --- Step 3: near start/finish detection ---

    dist_to_start = np.linalg.norm(car_position - centerline[0])
    near_start_finish = dist_to_start < 20.0

    # --- Step 4: smooth straightness measure ---
    inst_straight = 1.0 if is_straight else 0.0
    alpha_eff = 0.05 if inst_straight == 1.0 else 0.20
    smooth_straightness = (1 - alpha_eff) * smooth_straightness + alpha_eff * inst_straight
    # Compress straightness so car stays "curve-like" longer (bigger number = more curve-like)
    smooth_straightness = smooth_straightness ** 1.7
    smooth_straightness = np.clip(smooth_straightness, 0.0, 1.0)

    # blend
    curve_LA = 12.0
    straight_LA = 20.0
    base_lookahead = (1 - smooth_straightness) * curve_LA + smooth_straightness * straight_LA

    curve_gain = 0.15
    straight_gain = 0.25
    velocity_gain = (1 - smooth_straightness) * curve_gain + smooth_straightness * straight_gain

    if near_start_finish:
        base_lookahead = 10.0
        velocity_gain = 0.15
    lookahead_distance = base_lookahead + velocity_gain * abs(current_velocity)

    # --- Step 5: lookahead point for steering (r[i+1]) ---
    lookahead_idx = closest_idx
    accumulated_distance = 0.0
    while accumulated_distance < lookahead_distance:
        next_idx = (lookahead_idx + 1) % num_points
        segment = centerline[next_idx] - centerline[lookahead_idx]
        seg_len = np.linalg.norm(segment)
        accumulated_distance += seg_len
        lookahead_idx = next_idx
        if lookahead_idx == closest_idx and accumulated_distance > 0:
            break
    
    lookahead_point = centerline[lookahead_idx]

    # --- Step 6: assignment-based steering law (Question 2) ---
    dx = lookahead_point[0] - x
    dy = lookahead_point[1] - y
    phi_des_next = np.arctan2(dy, dx)

    heading_error = phi_des_next - heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    tau_delta = 0.05  # seconds
    v_r_model = np.clip(abs(current_velocity), 15.0, 40.0)
    delta_lin = (wheelbase / (v_r_model * tau_delta)) * heading_error

    k_delta = 0.5
    desired_steering = np.clip(k_delta * delta_lin, -float(parameters[4]), float(parameters[4]))

    # --- Step 7: curvature ahead for speed planning ---
    if current_velocity > 70 or tight_turn_hold > 0:
        reaction_distance = np.clip(2.5 * abs(current_velocity), 20.0, 120.0)
    else:
        reaction_distance = np.clip(abs(current_velocity), 10.0, 60.0)
    curv_start_idx = find_index_at_distance(centerline, closest_idx, reaction_distance)

    if current_velocity > 70 or tight_turn_hold > 0:
        num_ahead = 20
    else:
        num_ahead = 12

    speed_curvature = compute_curvature_lookahead(centerline, curv_start_idx, num_ahead)
    speed_curvature = max(speed_curvature, 0.0)

   # ------------------------------------
    # TIGHT TURN STATE MACHINE
    # ------------------------------------
    tight_turn_enter = 0.02      # enter hairpin mode
    tight_turn_exit  = 0.012      # require very low curvature to exit
    tight_turn_min_samples = 40   # how long to force slow mode minimum
    tight_turn_exit_stable = 12   # number of consecutive samples needed to exit

    # memory for stable exit counting
    if "tight_turn_exit_counter" not in globals():
        global tight_turn_exit_counter
        tight_turn_exit_counter = 0

    # ----- ENTER HAIRPIN MODE -----
    if speed_curvature > tight_turn_enter and current_velocity>70:
        # reset hold and exit counter
        tight_turn_hold = tight_turn_min_samples
        tight_turn_exit_counter = 0

    elif speed_curvature <= tight_turn_exit:
        tight_turn_exit_counter += 1
        if tight_turn_exit_counter >= tight_turn_exit_stable:
            # only now allow the hold to tick down
            if tight_turn_hold > 0:
                tight_turn_hold -= 1

    # --- Step 8: base curvature slowdown ---
    k_curv = 5.0
    base_speed = v_straight * np.exp(-k_curv * speed_curvature)
    target_velocity = base_speed

    # --- Step 8: boundary-based safety on speed ---

    right_distances = np.linalg.norm(racetrack.right_boundary - car_position, axis=1)
    left_distances  = np.linalg.norm(racetrack.left_boundary  - car_position, axis=1)
    min_boundary_dist = min(np.min(right_distances), np.min(left_distances))

    safety_threshold = 8.0 if not is_straight else 4.0
    if min_boundary_dist < safety_threshold:
        safety_factor = (min_boundary_dist / safety_threshold) ** 1.5
        target_velocity *= np.clip(safety_factor, 0.2, 1.0)

    # --- exponential slowdowns ---
    curv = speed_curvature
    curvature_factor = np.exp(-10.0 * curv)      # smaller weight
    target_velocity *= curvature_factor

    v = abs(current_velocity)
    speed_factor = np.exp(-0.005 * max(v - 35.0, 0.0))   # moderate weight
    target_velocity *= speed_factor

    h_err = abs(heading_error)
    heading_factor = np.exp(-3.5 * h_err)        # strongest effect
    target_velocity *= heading_factor

    # ---------------------------------------------
    # Unified Slow Mode:
    # - tight_turn_hold → sharp hairpins
    # Prevents early acceleration
    # ---------------------------------------------

    # print("tight ", tight_turn_hold)
    if tight_turn_hold > 0:
        desired_velocity = min(12.5, current_velocity, target_velocity)
    else:
        desired_velocity = np.clip(target_velocity, v_min, max_velocity)

    if desired_velocity < 0:
        desired_steering = -desired_steering

    return np.array([desired_steering, desired_velocity])