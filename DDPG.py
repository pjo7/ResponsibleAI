# good codde-------------------------------------------------------------------------------------code for displaying decisions
'''import carla
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import os
import random
from collections import deque
import pygame

# Hyperparameters
BUFFER_SIZE = 10
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
IMG_WIDTH, IMG_HEIGHT = 420,420
MAX_EPISODES = 1000
SAVE_PATH = "ddpg_carla_model"
DISPLAY_WIDTH=420
DISPLAY_HEIGHT=420
MAX_STEPS=500

# Create Actor Model
def create_actor():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=2)(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    steering = layers.Dense(1, activation='tanh')(x)
    acceleration = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, [steering, acceleration])

# Create Critic Model
def create_critic():
    state_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=2)(state_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=2)(x)
    x = layers.Flatten()(x)

    action_input = layers.Input(shape=(2,))
    action_out = layers.Dense(256, activation='relu')(action_input)

    concat = layers.Concatenate()([x, action_out])
    out = layers.Dense(256, activation='relu')(concat)
    outputs = layers.Dense(1)(out)

    return tf.keras.Model([state_input, action_input], outputs)

# Ornstein-Uhlenbeck noise
def ou_noise(x, mu=0, theta=0.15, sigma=0.2):
    return theta * (mu - x) + sigma * np.random.normal()

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), 
                np.array(actions, dtype=np.float32), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states, dtype=np.float32), 
                np.array(dones, dtype=np.float32))

def find_distance_to_destination(vehicle_location):
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y
    x_vh = vehicle_location.x
    y_vh = vehicle_location.y
    wp_array = np.array([x_wp,y_wp])
    vh_array = np.array([x_vh,y_vh])
    return np.linalg.norm(wp_array-vh_array)
    

    
# Define reward function
def calculate_reward(speed, acceleration, deviation,distance_to_destination,steps,collision):
    #speed_reward = speed / 30.0  
    #deviation_penalty = deviation
    #distance_penalty = -distance_to_destination
    #acceleration_penalty = -abs(acceleration) / 5.0
    #total_reward = deviation_penalty+distance_penalty
    #return total_reward
    # Parameters
    w_progress = 1.0
    w_safety = 1.5
    w_comfort = 0.5
    w_lane = 1.0
    w_traffic = 1.5
    
    # Progress Reward
    progress_reward = speed * max(0, deviation) * steps
    
    # Safety Reward
    if collision:
        safety_reward = -100
    else:
        min_distance = state['min_distance']
        safety_reward = -1 / min_distance if min_distance < state['safe_distance'] else 0
    
    # Comfort Reward
    comfort_reward = -(action['acceleration_delta']**2) - (action['steering_delta']**2)
    
    # Lane Keeping Reward
    lane_reward = -abs(lane_deviation)
    
    # Traffic Rules Reward
    if speed > speed_limit:
        traffic_reward = -10
    elif traffic_light == 'red' and speed > 0:
        traffic_reward = -50
    else:
        traffic_reward = 0
    
    # Total Reward
    total_reward = (w_progress * progress_reward +
                    w_safety * safety_reward +
                    w_comfort * comfort_reward +
                    w_lane * lane_reward +
                    w_traffic * traffic_reward)
    
    return total_reward

# Soft update
def update_target_weights(target, source, tau):
    for t, s in zip(target.trainable_variables, source.trainable_variables):
        t.assign(t * (1 - tau) + s * tau)
        
        
def calculate_angle_deviation(vehicle_yaw, road_yaw):
    deviation = np.cos((vehicle_yaw - road_yaw)*np.pi/180)
    return deviation

# Initialize CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.load_world('Town01')

# Initialize DDPG components
actor = create_actor()
critic = create_critic()
actor_target = create_actor()
critic_target = create_critic()

actor_target.set_weights(actor.get_weights())
critic_target.set_weights(critic.get_weights())

actor_optimizer = tf.keras.optimizers.Adam(LR_ACTOR)
critic_optimizer = tf.keras.optimizers.Adam(LR_CRITIC)

replay_buffer = ReplayBuffer(BUFFER_SIZE)

episode_rewards = []

# Initialize Pygame
pygame.init()
DISPLAY_WIDTH, DISPLAY_HEIGHT = 420,420
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption("CARLA DDPG Agent")
clock = pygame.time.Clock()
spawn_point =None
destination = None
# Training loop
for episode in range(MAX_EPISODES):
    # Reset environment and get initial state
    steps=0
    vehicle = None
    camera = None
    collision_sensor = None
    lane_sensor = None
    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        wmap = world.get_map()
        spawn_points = wmap.get_spawn_points()
        random.shuffle(spawn_points)
        if spawn_point == None :
            spawn_point = random.choice(spawn_points)  # choose from the given points......Random can also be applied
            destination = random.choice(spawn_points).location
         # fixed a permenant start and stop   
        vehicles = []
        for i in range(min(num_cars, len(spawn_points))):
            vehicle_bp = random.choice(vehicle_blueprints)  # Randomly choose a vehicle
            transform = spawn_points[i]  # Assign a spawn point
            if transform == spawn_point:
            	continue
            vehicle = world.try_spawn_actor(vehicle_bp, transform)
            if vehicle:
                vehicles.append(vehicle)
                vehicle.set_autopilot(True)  # Enable autopilot mode
                print(f"Spawned vehicle {vehicle.type_id} at {transform.location}")
        
                # --- Spawn Pedestrians ---
        num_pedestrians = 30
        pedestrian_blueprints = blueprint_library.filter('walker.pedestrian.*')  # Pedestrian blueprints
        pedestrian_controller_bp = blueprint_library.find('controller.ai.walker')  # Pedestrian AI controller

        pedestrians = []
        controllers = []
        for _ in range(num_pedestrians):
            pedestrian_bp = random.choice(pedestrian_blueprints)  # Randomly choose a pedestrian
            spawn_point = carla.Transform()
            spawn_point.location = world.get_random_location_from_navigation()  # Get random navigable location
            if spawn_point.location:
                pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
                if pedestrian:
                    pedestrians.append(pedestrian)
                    print(f"Spawned pedestrian {pedestrian.type_id} at {spawn_point.location}")

                    # Attach a controller to the pedestrian
                    controller = world.spawn_actor(pedestrian_controller_bp, carla.Transform(), attach_to=pedestrian)
                    controllers.append(controller)

        # --- Start Pedestrian Controllers ---
        for controller in controllers:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.4)  # Set walking speed for pedestrians

        print(f"Spawned {len(vehicles)} vehicles and {len(pedestrians)} pedestrians.")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(False)

        # Sensor setup (RGB camera)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMG_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        def process_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
            return array

        image_data = [None]
        camera.listen(lambda image: image_data.__setitem__(0, process_image(image)))

        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform()
        collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)

        collision_detected = [False]
        collision_sensor.listen(lambda event: collision_detected.__setitem__(0, True))

        # Lane invasion sensor
        lane_bp = blueprint_library.find('sensor.other.lane_invasion')
        lane_transform = carla.Transform()
        lane_sensor = world.spawn_actor(lane_bp, lane_transform, attach_to=vehicle)

        lane_deviation = [0.0]

        def lane_invasion_callback(event):
            lane_deviation[0] += 1.0

        lane_sensor.listen(lane_invasion_callback)

        state = None
        while state is None:
            world.tick()
            state = image_data[0]

        episode_reward = 0
        done = False
        # Initialize noise
        ou_noise_steering = 0
        ou_noise_acceleration = 0
        while not done :
            steps+=1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            world.tick()

            # Get current state
            speed = vehicle.get_velocity().length()
            acceleration = vehicle.get_acceleration().length()
            deviation = lane_deviation[0]
            vehicle_location = vehicle.get_location()
            vehicle_yaw = vehicle.get_transform().rotation.yaw

            # Get nearest waypoint
            waypoint = wmap.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            road_yaw = waypoint.transform.rotation.yaw
            angle_deviation = calculate_angle_deviation(vehicle_yaw, road_yaw)
            
            # Select action
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            steering, accel = actor(state_tensor)
            # Update action selection
            ou_noise_steering = ou_noise(ou_noise_steering)
            ou_noise_acceleration = ou_noise(ou_noise_acceleration)
            action = [steering.numpy()[0][0] + ou_noise_steering, accel.numpy()[0][0] + ou_noise_acceleration]
            print("Start location : ", spawn_point.location," Destination Location : ",destination)
            print(f"State: Current Speed={speed:.2f}, Acceleration={acceleration:.2f}, Deviation={angle_deviation:.2f}")
            print(f"Action: Steering={action[0]:.2f}, Acceleration={action[1]:.2f}")

            # Apply action
            vehicle.apply_control(carla.VehicleControl(steer=float(action[0]), throttle=float(action[1])))

            # Get next state
            next_state = image_data[0]
            
            #getting current data
            vehicle_location = vehicle.get_location()
            
            distance_to_destination = vehicle_location.distance(destination)
            print(f"Distance to destination : {distance_to_destination:.2f}")

            # Calculate reward
            reward = calculate_reward(speed, acceleration, angle_deviation,distance_to_destination,steps)

            # Check termination
            done = collision_detected[0] or distance_to_destination < 2.0
            if collision_detected[0]:
                reward-=1000
            if distance_to_destination < 2.0:
                reward+=10000

            # Store in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            print("Episode : ",episode," Reward : ",episode_reward)

            # Render the camera feed in Pygame
            if state is not None:
                frame = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                scaled_frame = pygame.transform.scale(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                screen.blit(scaled_frame, (0, 0))
                pygame.display.flip()

            clock.tick(30)  # Limit to 30 FPS

        episode_rewards.append(episode_reward)

        # Train model
        if len(replay_buffer.buffer) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # Preprocess tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # Critic update
            next_actions = actor_target(next_states)
            next_actions = tf.concat(next_actions, axis=1)  # Combine steering and acceleration
            target_q = rewards + GAMMA * (1 - dones) * tf.squeeze(critic_target([next_states, next_actions]), axis=1)
            with tf.GradientTape() as tape:
                q_values = tf.squeeze(critic([states, actions]), axis=1)
                critic_loss = tf.reduce_mean(tf.square(target_q - q_values))
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

            # Actor update
            with tf.GradientTape() as tape:
                current_actions = actor(states)
                current_actions = tf.concat(current_actions, axis=1)  # Combine steering and acceleration to form a single action group
                critic_value = -tf.reduce_mean(critic([states, current_actions]))
            actor_grads = tape.gradient(critic_value, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

            # Update target networks
            update_target_weights(actor_target, actor, TAU)
            update_target_weights(critic_target, critic_target, TAU)

    finally:
        if vehicle is not None:
            vehicle.destroy()
        if camera is not None:
            camera.destroy()
        if collision_sensor is not None:
            collision_sensor.destroy()
        if lane_sensor is not None:
            lane_sensor.destroy()
        # --- Cleanup ---
        print("Cleaning up actors...")
        for controller in controllers:
            controller.stop()
            controller.destroy()
        for pedestrian in pedestrians:
            pedestrian.destroy()
        for vehicle in vehicles:
            vehicle.destroy()
        print("Cleaned up actors.")

# Save the trained model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
actor.save(os.path.join(SAVE_PATH, 'actor.h5'))
critic.save(os.path.join(SAVE_PATH, 'critic.h5'))

# Plot episode rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')
plt.show()
pygame.quit()'''


#updated code
# good codde-------------------------------------------------------------------------------------code for displaying decisions
import carla
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import os
import random
from collections import deque
import pygame

# Hyperparameters
BUFFER_SIZE = 100
BATCH_SIZE = 5
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
IMG_WIDTH, IMG_HEIGHT = 420,420
MAX_EPISODES = 1000
SAVE_PATH = "ddpg_carla_model"
DISPLAY_WIDTH=420
DISPLAY_HEIGHT=420
MAX_STEPS=500

# Create Actor Model
def create_actor():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=2)(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    steering = layers.Dense(1, activation='tanh')(x)
    acceleration = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, [steering, acceleration])

# Create Critic Model
def create_critic():
    state_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=2)(state_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=2)(x)
    x = layers.Flatten()(x)

    action_input = layers.Input(shape=(2,))
    action_out = layers.Dense(256, activation='relu')(action_input)

    concat = layers.Concatenate()([x, action_out])
    out = layers.Dense(256, activation='relu')(concat)
    outputs = layers.Dense(1)(out)

    return tf.keras.Model([state_input, action_input], outputs)

# Ornstein-Uhlenbeck noise
def ou_noise(x, mu=0, theta=0.15, sigma=0.2):
    return theta * (mu - x) + sigma * np.random.normal()

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), 
                np.array(actions, dtype=np.float32), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states, dtype=np.float32), 
                np.array(dones, dtype=np.float32))

def find_distance_to_destination(vehicle_location):
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y
    x_vh = vehicle_location.x
    y_vh = vehicle_location.y
    wp_array = np.array([x_wp,y_wp])
    vh_array = np.array([x_vh,y_vh])
    return np.linalg.norm(wp_array-vh_array)
    

    
# Define reward function
def calculate_reward(speed, acceleration, deviation,distance_to_destination,steps,collision,min_dist, delta_speed,delta_steer):
    #speed_reward = speed / 30.0  
    '''deviation_penalty = deviation
    distance_penalty = -distance_to_destination
    #acceleration_penalty = -abs(acceleration) / 5.0
    total_reward = deviation_penalty+distance_penalty
    return total_reward'''
    # Parameters
    w_progress = 1.0
    w_safety = 1.5
    w_comfort = 0.5
    w_lane = 1.0
    
    # Progress Reward
    progress_reward = speed * max(0, deviation) * steps
    
    # Safety Reward
    if collision:
        safety_reward = -100
    else:
        min_distance = min_dist
        safety_reward = -1 / min_dist #if min_dist < 2.0 else 0
    
    # Comfort Reward
    comfort_reward = -(delta_acceleration**2) - (delta_steer**2)
    
    # Lane Keeping Reward
    lane_reward = -abs(deviation)
    
    # Traffic Rules Reward
    # if speed > speed_limit:
    #     traffic_reward = -10
    # elif traffic_light == 'red' and speed > 0:
    #     traffic_reward = -50
    # else:
    #     traffic_reward = 0
    
    # Total Reward
    total_reward = (w_progress * progress_reward +
                    w_safety * safety_reward +
                    w_comfort * comfort_reward +
                    w_lane * lane_reward)
    print("Progress Reward : ",progress_reward," safety reward = ",safety_reward," comfort reward = ",comfort_reward," total reward : ",total_reward)
    if distance_to_destination < 10.0:
        reward+=1000
        print("Reacheddddddd")
    
    return total_reward

# Soft update
def update_target_weights(target, source, tau):
    for t, s in zip(target.trainable_variables, source.trainable_variables):
        t.assign(t * (1 - tau) + s * tau)
        
        
def calculate_angle_deviation(vehicle_yaw, road_yaw):
    theta = vehicle_yaw - road_yaw
    deviation = np.cos((theta)*np.pi/180)
    return deviation

# Initialize CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.load_world('Town01')

# Initialize DDPG components
actor = create_actor()
critic = create_critic()
actor_target = create_actor()
critic_target = create_critic()

actor_target.set_weights(actor.get_weights())
critic_target.set_weights(critic.get_weights())

actor_optimizer = tf.keras.optimizers.Adam(LR_ACTOR)
critic_optimizer = tf.keras.optimizers.Adam(LR_CRITIC)

replay_buffer = ReplayBuffer(BUFFER_SIZE)

episode_rewards = []

# Initialize Pygame
pygame.init()
DISPLAY_WIDTH, DISPLAY_HEIGHT = 420,420
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption("CARLA DDPG Agent")
clock = pygame.time.Clock()
start_spawn_point =None
destination = None
delta_acceleration=0
delta_steer=0
old_accel = 0
old_steer = 0
# Training loop
for episode in range(MAX_EPISODES):
    # Reset environment and get initial state
    steps=0
    vehicle = None
    camera = None
    collision_sensor = None
    lane_sensor = None
    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        wmap = world.get_map()
        spawn_points = wmap.get_spawn_points()
        random.shuffle(spawn_points)
        if start_spawn_point == None :
            start_spawn_point = random.choice(spawn_points)  # choose from the given points......Random can also be applied
            destination = random.choice(spawn_points).location
            print("Start location : ", start_spawn_point.location," Destination Location : ",destination)
         # fixed a permenant start and stop   
        '''vehicles = []
        num_cars=10
        for i in range(min(num_cars, len(spawn_points))):
            transform = spawn_points[i]  # Assign a spawn point
            if transform == start_spawn_point:
            	continue
            trial_vehicle = world.try_spawn_actor(vehicle_bp, transform)
            if trial_vehicle:
                vehicles.append(trial_vehicle)
                trial_vehicle.set_autopilot(True)  # Enable autopilot mode
                print(f"Spawned vehicle {trial_vehicle.type_id} at {transform.location}")
        
                # --- Spawn Pedestrians ---
        num_pedestrians = 10
        pedestrian_blueprints = blueprint_library.filter('walker.pedestrian.*')  # Pedestrian blueprints
        pedestrian_controller_bp = blueprint_library.find('controller.ai.walker')  # Pedestrian AI controller

        pedestrians = []
        controllers = []
        for _ in range(num_pedestrians):
            pedestrian_bp = random.choice(pedestrian_blueprints)  # Randomly choose a pedestrian
            spawn_point = carla.Transform()
            spawn_point.location = world.get_random_location_from_navigation()  # Get random navigable location
            if spawn_point.location:
                pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
                if pedestrian:
                    pedestrians.append(pedestrian)
                    print(f"Spawned pedestrian {pedestrian.type_id} at {spawn_point.location}")

                    # Attach a controller to the pedestrian
                    controller = world.spawn_actor(pedestrian_controller_bp, carla.Transform(), attach_to=pedestrian)
                    controllers.append(controller)

        # --- Start Pedestrian Controllers ---
        for controller in controllers:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.4)  # Set walking speed for pedestrians

        print(f"Spawned {len(vehicles)} vehicles and {len(pedestrians)} pedestrians.")'''
        vehicle = world.spawn_actor(vehicle_bp, start_spawn_point)
        vehicle.set_autopilot(True)
        print("Created ego vehicle")

        # Sensor setup (RGB camera)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMG_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        def process_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
            return array

        image_data = [None]
        camera.listen(lambda image: image_data.__setitem__(0, process_image(image)))

        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform()
        collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)

        collision_detected = [False]
        collision_sensor.listen(lambda event: collision_detected.__setitem__(0, True))

        # Lane invasion sensor
        # lane_bp = blueprint_library.find('sensor.other.lane_invasion')
        # lane_transform = carla.Transform()
        # lane_sensor = world.spawn_actor(lane_bp, lane_transform, attach_to=vehicle)

        # lane_deviation = [0.0]

        # def lane_invasion_callback(event):
        #     lane_deviation[0] += 1.0

        # lane_sensor.listen(lane_invasion_callback)

        state = None
        while state is None:
            world.tick()
            state = image_data[0]

        episode_reward = 0
        done = False
        # Initialize noise
        ou_noise_steering = 0
        ou_noise_acceleration = 0
        while not done :
            steps+=1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            world.tick()

            # Get current state
            
            speed = vehicle.get_velocity().length()
            acceleration = vehicle.get_acceleration().length()
            

            
            
            # deviation = lane_deviation[0]
            vehicle_location = vehicle.get_location()
            vehicle_yaw = vehicle.get_transform().rotation.yaw

            # Get nearest waypoint
            waypoint = wmap.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            
            road_yaw = waypoint.transform.rotation.yaw
            angle_deviation = calculate_angle_deviation(vehicle_yaw, road_yaw)
            
            
            # Select action
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            
            steering, accel = actor(state_tensor)
            #print("Getting velocity")
            
            # Update action selection
            ou_noise_steering = ou_noise(ou_noise_steering)
            ou_noise_acceleration = ou_noise(ou_noise_acceleration)
            action = [steering.numpy()[0][0] + ou_noise_steering, accel.numpy()[0][0] + ou_noise_acceleration]
            

            print(f"State: Current Speed={speed:.2f}, Acceleration={acceleration:.2f}, Deviation={angle_deviation:.2f}")
            print(f"Action: Steering={action[0]:.2f}, Acceleration={action[1]:.2f}")
            # calculate delta values for both
            delta_acceleration = acceleration-old_accel
            
            delta_steer = action[0] - old_steer
            
            # Apply action
            
            #vehicle.apply_control(carla.VehicleControl(steer=float(action[0]), throttle=float(action[1])))
            # Render the camera feed in Pygame
            if state is not None:
                frame = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                scaled_frame = pygame.transform.scale(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                screen.blit(scaled_frame, (0, 0))
                pygame.display.flip()

            clock.tick(30)  # Limit to 30 FPS

            # Get next state
            next_state = image_data[0]
            
            #getting current data
            vehicle_location = vehicle.get_location()
            
            distance_to_destination = vehicle_location.distance(destination)
            print(f"Distance to destination : {distance_to_destination:.2f}")

            # Calculate reward
            '''actor_list = world.get_actors()
            relevant_actors = actor_list.filter('vehicle.*') #+ actor_list.filter('walker.pedestrian.*')

            min_distance = float('inf')  # Initialize with a very large number
            for actor in relevant_actors:
                if actor.id != vehicle.id:  # Skip the ego vehicle
                    other_location = actor.get_transform().location
                    distance = vehicle_location.distance(other_location)  # Euclidean distance
                    if distance < min_distance:
                        min_distance = distance'''
            min_distance=2.0
            reward = calculate_reward(speed, acceleration, angle_deviation,distance_to_destination,steps,collision_detected[0],min_distance,delta_acceleration,delta_steer)

            # Check termination
            done = collision_detected[0] or distance_to_destination < 2.0
            

            # Store in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            old_accel = acceleration
            old_steer = action[0]
            

            episode_reward += reward
            print("Episode : ",episode," Reward : ",episode_reward)

            

        episode_rewards.append(episode_reward)

        # Train model
        if len(replay_buffer.buffer) > BATCH_SIZE:
            print("Updating weights........................................")
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # Preprocess tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # Critic update
            next_actions = actor_target(next_states)
            next_actions = tf.concat(next_actions, axis=1)  # Combine steering and acceleration
            target_q = rewards + GAMMA * (1 - dones) * tf.squeeze(critic_target([next_states, next_actions]), axis=1)
            with tf.GradientTape() as tape:
                q_values = tf.squeeze(critic([states, actions]), axis=1)
                critic_loss = tf.reduce_mean(tf.square(target_q - q_values))
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

            # Actor update
            with tf.GradientTape() as tape:
                current_actions = actor(states)
                current_actions = tf.concat(current_actions, axis=1)  # Combine steering and acceleration to form a single action group
                critic_value = -tf.reduce_mean(critic([states, current_actions]))
            actor_grads = tape.gradient(critic_value, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

            # Update target networks
            update_target_weights(actor_target, actor, TAU)
            update_target_weights(critic_target, critic_target, TAU)
    except Exception as e:
        print(e)

    finally:
        if vehicle is not None:
            vehicle.destroy()
        if camera is not None:
            camera.destroy()
        if collision_sensor is not None:
            collision_sensor.destroy()
        # if lane_sensor is not None:
        #     lane_sensor.destroy()
        # --- Cleanup ---
        print("Cleaning up actors...")
        '''for controller in controllers:
            print("destroying controllers")
        
            controller.stop()
            controller.destroy()
        for pedestrian in pedestrians:
            print("destroying peds")
            pedestrian.destroy()
        for vehicle in vehicles:
            print("destroying vehs")
            vehicle.destroy()'''
        print("Cleaned up actors.")

# Save the trained model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
actor.save(os.path.join(SAVE_PATH, 'actor.h5'))
critic.save(os.path.join(SAVE_PATH, 'critic.h5'))

# Plot episode rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')
plt.show()
pygame.quit()
