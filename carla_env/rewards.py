import numpy as np
from config import CONFIG

low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
reward_functions = {}


emotion_weights = {
    "C": 0.5,   # Calmness weight
    "Rt": -0.3,  # Restlessness weight (penalizes erratic driving)
    "U": 0.4,   # Urgency weight
    "L": -0.2   # Laziness weight (penalizes idling)
}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            # Stop if speed is less than 1.0 km/h after the first 5s of an episode
            global low_speed_timer
            low_speed_timer += 1.0 / env.fps
            speed = env.vehicle.get_speed()
            if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1:
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"

            # Stop if distance from center > max distance
            if env.distance_from_center > max_distance:
                env.terminal_state = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if max_speed > 0 and speed > max_speed:
                env.terminal_state = True
                terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func


# Reward_fn5
def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
               * distance_std_factor (1 when std from center lane is low, 0 when not)
    """

    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor * distance_std_factor

    L,R,U=reward_fn_emotion(env)
    reward += emotion_weights["U"]*U
    return reward


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


def reward_fn_waypoints(env):
    """
        reward
            - Each time the vehicle overpasses a waypoint, it will receive a reward of 1.0
            - When the vehicle does not pass a waypoint, it receives a reward of 0.0
    """
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]
    print("waypoint rew fn")
    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    reward = (env.current_waypoint_index - env.prev_waypoint_index) + speed_reward * centering_factor
    return reward


reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)

def reward_fn_emotion(env):
    step_count=max(env.step_count, 1)
    delta_v_avg = env.delta_v_accum / step_count
    delta_v_rel = env.delta_v_accum / step_count
    D_lane = max(env.distance_from_center, 0.1)
    a_avg = env.delta_v_accum / step_count
    d_goal = env.distance_to_goal - env.distance_traveled
    d_total = env.distance_to_goal
    ttc=env.last_ttc
    w1 = 0.1
    w2 = 0.2
    w3 = 0.3

    mu = 0.3
    delta_1, delta_2, delta_3 = 0.1, 0.2, 0.3
    lambda1 = 1.5  # Time to Collision weight
    lambda2 = 2.0  # Relative speed weight
    lambda3 = 3.0  # Pedestrian distance weight
    lambda4 = 2.5  # Traffic severity weight
    R_prev=getattr(env,"restless_prev",0)
    
    #print(f"step_count:{step_count} delta_v_avg:{delta_v_avg} d_total:{d_total} a_avg:{a_avg}")
     #Lazyness
    '''
    reward =  1-delta_v_avg/v_max * w1 + d_goal/d_total * w2 + (1-a_avg) * w3 + t_idle
    '''
    lazy_score = (1-delta_v_avg/max_speed) * w1 + d_goal/d_total * w2 + (1-a_avg) * w3
    restless_score= (1 - mu) * R_prev + mu * (delta_1 * delta_v_rel + delta_2 * (1 / D_lane)) #+ delta_3 * S_traffic)
    urgency_score = (lambda1 *(1/(ttc+ 1e-6))) + (lambda2 * abs(delta_v_rel)) #+ (lambda3 * d_ped) + (lambda4 * s_traffic)
    urgency_score = np.clip(urgency_score, -1, 1)
    env.restless_prev=restless_score
    
    return lazy_score, restless_score, urgency_score
    
reward_functions["reward_fn_emotion"] = create_reward_fn(reward_fn_emotion)
