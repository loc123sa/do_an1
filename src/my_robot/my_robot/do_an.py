import rclpy
from rclpy.node import Node
 
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
 
import numpy as np
import math
from collections import deque
from typing import List, Tuple, Optional
 
 
class ImprovedStableDWADynamic(Node):
   
    def __init__(self):
        super().__init__('improved_stable_dwa_dynamic')
 
        # SUBSCRIBERS
        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
 
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
 
        # PUBLISHER
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
 
        # CURRENT STATE
        self.scan_data = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_scan_time = None
 
        # GOAL
        self.goal_x = 5.0
        self.goal_y = 0.0
 
        # DWA PARAMETERS - ADJUSTED FOR DYNAMIC ENVIRONMENTS
        self.max_speed = 0.20
        self.max_yawrate = 0.8
        self.v_resolution = 0.04
        self.w_resolution = 0.2
        self.predict_time = 2.0  # ⬆️ INCREASED from 1.0 → 2.0 for dynamic obstacles
        
        # OBSTACLE AVOIDANCE PARAMETERS - TIGHTENED FOR SAFETY
        self.min_safe_distance = 0.60  # ⬆️ INCREASED from 0.45 → 0.60
        self.critical_distance = 0.25
        self.danger_distance = 0.35
        
        # PATH HISTORY TRACKING
        self.path_history = deque(maxlen=100)
        self.history_position_threshold = 0.1
        self.loop_detection_radius = 0.5
        
        # NEW: DYNAMIC OBSTACLE TRACKING
        self.obstacle_history = deque(maxlen=50)  # Store last 50 scan frames
        self.obstacle_positions = {}  # ID -> list of (x, y, t) positions
        self.obstacle_velocities = {}  # ID -> (vx, vy) estimated velocity
        self.last_frame_obstacles = None
        
        # COST FUNCTION WEIGHTS - ADJUSTED FOR DYNAMIC
        self.weight_goal = 1.2  # DECREASED from 1.5 → 1.2
        self.weight_obstacle = 3.5  # INCREASED from 2.0 → 3.5
        self.weight_turn = 0.4
        self.weight_speed = 0.2
        self.weight_path_history = 0.3  # DECREASED from 0.8 → 0.3
        self.weight_dynamic_collision = 1.5  # NEW weight for predicted collisions
        
        # TIMER
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('IMPROVED DWA FOR DYNAMIC MAP STARTED')
 
    def scan_callback(self, msg):
        """Store laser scan data and track obstacles"""
        self.scan_data = np.array(msg.ranges)
        self.last_scan_time = self.get_clock().now().nanoseconds / 1e9
        
        #  NEW: Extract and track obstacle positions
        self.update_obstacle_history()
 
    def odom_callback(self, msg):
        """Extract position and orientation from odometry"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
 
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Store position in history
        if len(self.path_history) == 0 or \
           math.sqrt((self.x - self.path_history[-1][0])**2 + 
                    (self.y - self.path_history[-1][1])**2) > self.history_position_threshold:
            self.path_history.append((self.x, self.y))
 
    #  NEW FUNCTIONS FOR DYNAMIC OBSTACLE HANDLING
    
    def update_obstacle_history(self):
      
        if self.scan_data is None:
            return
        
        current_obstacles = []
        
        # Convert polar coordinates (laser) to cartesian coordinates
        for angle_idx, range_val in enumerate(self.scan_data):
            if not np.isfinite(range_val) or range_val < 0.1:
                continue
            
            # Calculate angle in world frame
            angle = (angle_idx / len(self.scan_data)) * 2 * math.pi + self.yaw
            
            # Convert to world coordinates relative to robot
            obs_x = self.x + range_val * math.cos(angle)
            obs_y = self.y + range_val * math.sin(angle)
            
            current_obstacles.append((obs_x, obs_y, range_val))
        
        self.last_frame_obstacles = current_obstacles
        self.obstacle_history.append((self.last_scan_time, current_obstacles))
        
        # Estimate velocities if we have history
        if len(self.obstacle_history) >= 2:
            self.estimate_obstacle_velocities()
 
    def estimate_obstacle_velocities(self):
       
        if len(self.obstacle_history) < 2:
            return
        
        time_prev, obs_prev = self.obstacle_history[-2]
        time_curr, obs_curr = self.obstacle_history[-1]
        
        dt = time_curr - time_prev
        if dt < 0.001:  # Avoid division by very small numbers
            return
        
        # Simple clustering: match obstacles from previous frame to current frame
        for i, (obs_x_curr, obs_y_curr, range_curr) in enumerate(obs_curr):
            # Find closest obstacle in previous frame
            min_dist = float('inf')
            closest_idx = -1
            
            for j, (obs_x_prev, obs_y_prev, range_prev) in enumerate(obs_prev):
                dist = math.sqrt((obs_x_curr - obs_x_prev)**2 + 
                               (obs_y_curr - obs_y_prev)**2)
                if dist < min_dist and dist < 0.5:  # Max matching distance
                    min_dist = dist
                    closest_idx = j
            
            # Calculate velocity if match found
            if closest_idx >= 0:
                obs_x_prev, obs_y_prev, _ = obs_prev[closest_idx]
                vx = (obs_x_curr - obs_x_prev) / dt
                vy = (obs_y_curr - obs_y_prev) / dt
                
                # Store as average with exponential weighting
                obs_id = i
                if obs_id not in self.obstacle_velocities:
                    self.obstacle_velocities[obs_id] = (vx, vy)
                else:
                    old_vx, old_vy = self.obstacle_velocities[obs_id]
                    # Smooth velocity estimates
                    alpha = 0.3
                    self.obstacle_velocities[obs_id] = (
                        alpha * vx + (1 - alpha) * old_vx,
                        alpha * vy + (1 - alpha) * old_vy
                    )
 
    def predict_obstacle_position(self, obs_x: float, obs_y: float, 
                                 obs_id: int, predict_time: float) -> Tuple[float, float]:
        
        if obs_id in self.obstacle_velocities:
            vx, vy = self.obstacle_velocities[obs_id]
            
            # Simple linear extrapolation: p_future = p_current + v * t
            future_x = obs_x + vx * predict_time
            future_y = obs_y + vy * predict_time
            
            return future_x, future_y
        else:
            # No velocity estimate yet, assume stationary
            return obs_x, obs_y
 
    def check_trajectory_collision_dynamic(self, trajectory: List[Tuple]) -> Tuple[float, float]:
        
        if self.scan_data is None:
            return 999, 0
        
        min_dist = 999
        collision_risk = 0
        
        # Check collision with static obstacles (laser scan)
        for traj_x, traj_y in trajectory:
            dx = traj_x - self.x
            dy = traj_y - self.y
            
            distance_from_origin = math.sqrt(dx**2 + dy**2)
            if distance_from_origin > self.predict_time * self.max_speed * 1.5:
                continue
            
            if len(self.scan_data) > 0:
                valid_ranges = self.scan_data[np.isfinite(self.scan_data)]
                if len(valid_ranges) > 0:
                    min_scan_dist = np.min(valid_ranges)
                    min_dist = min(min_dist, min_scan_dist)
        
        # NEW: Check collision with PREDICTED dynamic obstacles
        if self.last_frame_obstacles is not None:
            for obs_idx, (obs_x, obs_y, range_val) in enumerate(self.last_frame_obstacles):
                # Predict where this obstacle will be
                for traj_point_idx, (traj_x, traj_y) in enumerate(trajectory):
                    # Time in trajectory
                    t_traj = traj_point_idx * 0.1  # 0.1s per step
                    
                    # Predict obstacle position at this time
                    pred_obs_x, pred_obs_y = self.predict_obstacle_position(
                        obs_x, obs_y, obs_idx, t_traj
                    )
                    
                    # Distance from trajectory point to predicted obstacle
                    dist_to_obs = math.sqrt((traj_x - pred_obs_x)**2 + 
                                           (traj_y - pred_obs_y)**2)
                    
                    min_dist = min(min_dist, dist_to_obs)
                    
                    # Add collision risk if too close to predicted position
                    if dist_to_obs < self.danger_distance:
                        collision_risk += (self.danger_distance - dist_to_obs) * 15
        
        return min_dist, collision_risk
 
    # END NEW FUNCTIONS 
 
    def simulate_trajectory(self, v, w, dt=0.1):
        """
        Simulate trajectory for given linear and angular velocities
        Returns both final position and intermediate points for collision checking
        """
        x = self.x
        y = self.y
        yaw = self.yaw
        
        trajectory = [(x, y)]
        
        steps = int(self.predict_time / dt)
        for _ in range(steps):
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += w * dt
            trajectory.append((x, y))
 
        return x, y, yaw, trajectory
 
    def goal_cost(self, x, y):
        """Distance to goal"""
        return math.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
 
    def obstacle_cost(self, min_distance):
        """
        Obstacle avoidance cost with exponential gradient
        More aggressive for dynamic environments
        """
        if min_distance < self.critical_distance:
            return 999  # Reject dangerous trajectories
        
        if min_distance < self.danger_distance:
            # Steeper gradient for dynamic environments
            return 80 * (1.0 / (min_distance + 0.01))
        
        if min_distance < self.min_safe_distance:
            return 20 * (1.0 / (min_distance + 0.01))
        
        return 0
 
    def path_history_cost(self, x, y):
        
        if len(self.path_history) < 5:
            return 0
        
        penalty = 0
        
        # Only check most recent positions (last 10 instead of 20)
        for hist_x, hist_y in list(self.path_history)[-10:]:
            dist = math.sqrt((x - hist_x)**2 + (y - hist_y)**2)
            
            if dist < self.loop_detection_radius:
                penalty += 2.0 * math.exp(-dist / self.loop_detection_radius)  # Reduced
        
        return penalty
 
    def get_obstacle_metrics(self):
       
        if self.scan_data is None:
            return None, None, None, None, None
 
        scan_len = len(self.scan_data)
        front_center = self.scan_data[0:25]
        front_left = self.scan_data[25:80]
        front_right = self.scan_data[-80:-25]
        left = self.scan_data[80:140]
        right = self.scan_data[-140:-80]
 
        front_center = front_center[np.isfinite(front_center)]
        front_left = front_left[np.isfinite(front_left)]
        front_right = front_right[np.isfinite(front_right)]
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
 
        if len(front_center) == 0:
            return None, None, None, None, None
 
        front_min = np.min(front_center)
        front_left_min = np.min(front_left) if len(front_left) > 0 else 999
        front_right_min = np.min(front_right) if len(front_right) > 0 else 999
        left_mean = np.mean(left) if len(left) > 0 else 0.0
        right_mean = np.mean(right) if len(right) > 0 else 0.0
 
        return front_min, front_left_min, front_right_min, left_mean, right_mean
 
    def emergency_avoidance(self, front_min, front_left_min, front_right_min, 
                           left_mean, right_mean):
        """
        Emergency obstacle avoidance - more aggressive for dynamic maps
        """
        cmd = Twist()
 
        if (front_min < self.critical_distance or
            front_left_min < 0.25 or
            front_right_min < 0.25):
 
            cmd.linear.x = 0.0
            
            if left_mean > right_mean:
                cmd.angular.z = 0.8  # Faster turning
            else:
                cmd.angular.z = -0.8
            
            self.cmd_pub.publish(cmd)
            self.get_logger().warn(' EMERGENCY AVOIDANCE ACTIVATED')
            return True
        
        if (front_min < self.danger_distance or
            front_left_min < 0.30 or
            front_right_min < 0.30):
            
            cmd.linear.x = 0.03  # Even slower for dynamic
            
            if left_mean > right_mean:
                cmd.angular.z = 0.4
            else:
                cmd.angular.z = -0.4
            
            self.cmd_pub.publish(cmd)
            self.get_logger().info(' HIGH CAUTION MODE')
            return True
 
        return False
 
    def control_loop(self):
        
        if self.scan_data is None:
            return
 
        front_min, front_left_min, front_right_min, left_mean, right_mean = \
            self.get_obstacle_metrics()
 
        if front_min is None:
            return
 
        # Emergency avoidance check
        if self.emergency_avoidance(front_min, front_left_min, front_right_min, 
                                    left_mean, right_mean):
            return
 
        # Adaptive weights for dynamic situations
        if front_min < 0.6:
            self.weight_obstacle = 4.0  # Very high for close obstacles
            self.weight_goal = 0.8
        elif front_min < 1.0:
            self.weight_obstacle = 3.5
            self.weight_goal = 1.0
        else:
            self.weight_obstacle = 2.5
            self.weight_goal = 1.2
 
        # DWA SEARCH
        best_cost = 999999
        best_v = 0.08
        best_w = 0.0
 
        for v in np.arange(0.05, self.max_speed, self.v_resolution):
            for w in np.arange(-self.max_yawrate, self.max_yawrate, self.w_resolution):
                
                # Simulate trajectory
                tx, ty, tyaw, trajectory = self.simulate_trajectory(v, w)
 
                # Calculate costs
                g_cost = self.goal_cost(tx, ty)
                
                # Use enhanced collision checking
                min_traj_dist, collision_risk = self.check_trajectory_collision_dynamic(trajectory)
                obs_cost = self.obstacle_cost(min_traj_dist) + collision_risk
                
                # Path history with reduced impact
                hist_cost = self.path_history_cost(tx, ty)
                
                # Penalize sharp turns
                turn_cost = abs(w)
                
                # Prefer speed when safe
                speed_cost = (self.max_speed - v) if min_traj_dist > self.danger_distance else 0
                
                # Combine all costs
                total_cost = (
                    self.weight_goal * g_cost +
                    self.weight_obstacle * obs_cost +
                    self.weight_path_history * hist_cost +
                    self.weight_turn * turn_cost +
                    self.weight_speed * speed_cost
                )
 
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_v = v
                    best_w = w
 
        # PUBLISH COMMAND
        cmd = Twist()
        cmd.linear.x = float(best_v)
        cmd.angular.z = float(best_w)
        self.cmd_pub.publish(cmd)
        
        # Log occasionally
        if int(self.get_clock().now().nanoseconds / 1e9) % 2 == 0:
            self.get_logger().info(
                f'v={best_v:.2f}, w={best_w:.2f}, front={front_min:.2f}m, '
                f'obs_tracked={len(self.obstacle_velocities)}'
            )
 
 
def main(args=None):
    rclpy.init(args=args)
    node = ImprovedStableDWADynamic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
