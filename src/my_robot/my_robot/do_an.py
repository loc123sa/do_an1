
import rclpy
from rclpy.node import Node
 
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
 
import numpy as np
import math
from collections import deque
 
 
class ImprovedStableDWA(Node):
    """
    Enhanced DWA (Dynamic Window Approach) with:
    - Better obstacle avoidance using gradient cost
    - Path history tracking to avoid repetition
    - Collision checking along entire trajectory
    - Dynamic weight adjustment based on situation
    """
 
    def __init__(self):
        super().__init__('improved_stable_dwa')
 
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
 
        # GOAL
        self.goal_x = 5.0
        self.goal_y = 0.0
 
        # DWA PARAMETERS
        self.max_speed = 0.20
        self.max_yawrate = 0.8
        self.v_resolution = 0.04
        self.w_resolution = 0.2
        self.predict_time = 1.0
        
        # OBSTACLE AVOIDANCE PARAMETERS
        self.min_safe_distance = 0.45
        self.critical_distance = 0.25
        self.danger_distance = 0.35
        
        # PATH HISTORY TRACKING
        # Store recent positions to detect and avoid loops
        self.path_history = deque(maxlen=100)
        self.history_position_threshold = 0.1  # min distance between stored positions
        self.loop_detection_radius = 0.5  # radius to check for recent positions
        
        # COST FUNCTION WEIGHTS
        self.weight_goal = 1.5
        self.weight_obstacle = 2.0
        self.weight_turn = 0.4
        self.weight_speed = 0.2
        self.weight_path_history = 0.8
        
        # TIMER
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('IMPROVED STABLE DWA STARTED')
 
    def scan_callback(self, msg):
        """Store laser scan data"""
        self.scan_data = np.array(msg.ranges)
 
    def odom_callback(self, msg):
        """Extract position and orientation from odometry"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
 
        q = msg.pose.pose.orientation
 
        # Convert quaternion to yaw angle
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Store position in history periodically
        if len(self.path_history) == 0 or \
           math.sqrt((self.x - self.path_history[-1][0])**2 + 
                    (self.y - self.path_history[-1][1])**2) > self.history_position_threshold:
            self.path_history.append((self.x, self.y))
 
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
 
    def check_trajectory_collision(self, trajectory):
        """
        Check if trajectory collides with obstacles
        Returns minimum distance encountered along trajectory and collision risk score
        """
        if self.scan_data is None:
            return 999, 0
        
        min_dist = 999
        collision_risk = 0
        
        # Check each point in trajectory against laser data
        for traj_x, traj_y in trajectory:
            # Calculate angle from robot origin to trajectory point
            dx = traj_x - self.x
            dy = traj_y - self.y
            
            # Only check if point is reasonably close
            distance_from_origin = math.sqrt(dx**2 + dy**2)
            if distance_from_origin > self.predict_time * self.max_speed * 1.5:
                continue
            
            # Get closest obstacle in scan data
            if len(self.scan_data) > 0:
                valid_ranges = self.scan_data[np.isfinite(self.scan_data)]
                if len(valid_ranges) > 0:
                    min_scan_dist = np.min(valid_ranges)
                    min_dist = min(min_dist, min_scan_dist)
                    
                    # Add collision risk for points too close
                    if min_scan_dist < self.danger_distance:
                        collision_risk += (self.danger_distance - min_scan_dist) * 10
 
        return min_dist, collision_risk
 
    def goal_cost(self, x, y):
        """Distance to goal"""
        return math.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)
 
    def obstacle_cost(self, min_distance):
        """
        Obstacle avoidance cost with gradient
        Exponential penalty as we get closer to obstacles
        """
        if min_distance < self.critical_distance:
            return 999  # Reject dangerous trajectories
        
        if min_distance < self.danger_distance:
            # Steep gradient penalty in danger zone
            return 50 * (1.0 / (min_distance + 0.01))
        
        if min_distance < self.min_safe_distance:
            # Medium penalty in caution zone
            return 10 * (1.0 / (min_distance + 0.01))
        
        return 0  # Safe zone
 
    def path_history_cost(self, x, y):
        """
        Penalize trajectories that lead to recently visited areas
        This helps avoid loops and repetitive paths
        """
        if len(self.path_history) < 5:
            return 0
        
        penalty = 0
        
        # Check distance to recent positions
        for hist_x, hist_y in list(self.path_history)[-20:]:  # Check last 20 positions
            dist = math.sqrt((x - hist_x)**2 + (y - hist_y)**2)
            
            if dist < self.loop_detection_radius:
                # Exponential penalty for being in recently visited area
                penalty += 5.0 * math.exp(-dist / self.loop_detection_radius)
        
        return penalty
 
    def get_obstacle_metrics(self):
        """
        Extract detailed obstacle information from laser scan
        Returns metrics for emergency avoidance logic
        """
        if self.scan_data is None:
            return None, None, None, None, None
 
        # Define scan sectors
        scan_len = len(self.scan_data)
        front_center = self.scan_data[0:25]
        front_left = self.scan_data[25:80]
        front_right = self.scan_data[-80:-25]
        left = self.scan_data[80:140]
        right = self.scan_data[-140:-80]
 
        # Remove invalid values
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
 
    def emergency_avoidance(self, front_min, front_left_min, front_right_min, left_mean, right_mean):
        """
        Emergency obstacle avoidance when obstacles are too close
        Returns True if emergency maneuver was executed
        """
        cmd = Twist()
 
        # Trigger emergency stop and turn
        if (front_min < self.critical_distance or
            front_left_min < 0.25 or
            front_right_min < 0.25):
 
            cmd.linear.x = 0.0
            
            # Choose direction with more space
            if left_mean > right_mean:
                cmd.angular.z = 0.6
            else:
                cmd.angular.z = -0.6
            
            self.cmd_pub.publish(cmd)
            self.get_logger().warn('EMERGENCY AVOIDANCE ACTIVATED')
            return True
        
        # High caution mode - reduce speed and prepare to turn
        if (front_min < self.danger_distance or
            front_left_min < 0.30 or
            front_right_min < 0.30):
            
            cmd.linear.x = 0.05  # Very slow
            
            if left_mean > right_mean:
                cmd.angular.z = 0.3
            else:
                cmd.angular.z = -0.3
            
            self.cmd_pub.publish(cmd)
            self.get_logger().info('HIGH CAUTION MODE')
            return True
 
        return False
 
    def control_loop(self):
        """Main control loop - DWA algorithm with improvements"""
        if self.scan_data is None:
            return
 
        # Get obstacle metrics
        front_min, front_left_min, front_right_min, left_mean, right_mean = \
            self.get_obstacle_metrics()
 
        if front_min is None:
            return
 
        # Check if emergency avoidance is needed
        if self.emergency_avoidance(front_min, front_left_min, front_right_min, left_mean, right_mean):
            return
 
        # Adjust weights based on proximity to obstacles
        if front_min < 0.6:
            self.weight_obstacle = 3.0  # Increase obstacle avoidance
            self.weight_goal = 1.0      # Decrease goal seeking
        else:
            self.weight_obstacle = 2.0
            self.weight_goal = 1.5
 
        # DWA SEARCH
        best_cost = 999999
        best_v = 0.12
        best_w = 0.0
 
        for v in np.arange(0.08, self.max_speed, self.v_resolution):
            for w in np.arange(-self.max_yawrate, self.max_yawrate, self.w_resolution):
                
                # Simulate trajectory
                tx, ty, tyaw, trajectory = self.simulate_trajectory(v, w)
 
                # Calculate costs
                g_cost = self.goal_cost(tx, ty)
                
                # Check trajectory for collisions
                min_traj_dist, collision_risk = self.check_trajectory_collision(trajectory)
                obs_cost = self.obstacle_cost(min_traj_dist) + collision_risk
                
                # Path history cost - avoid looping
                hist_cost = self.path_history_cost(tx, ty)
                
                # Prefer straighter paths
                turn_cost = abs(w)
                
                # Prefer faster forward movement (but not if dangerous)
                speed_cost = (self.max_speed - v) if min_traj_dist > self.danger_distance else 0
                
                # Combine all costs with weighted sum
                total_cost = (
                    self.weight_goal * g_cost +
                    self.weight_obstacle * obs_cost +
                    self.weight_path_history * hist_cost +
                    self.weight_turn * turn_cost +
                    self.weight_speed * speed_cost
                )
 
                # Update best trajectory
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_v = v
                    best_w = w
 
        # PUBLISH COMMAND
        cmd = Twist()
        cmd.linear.x = float(best_v)
        cmd.angular.z = float(best_w)
        self.cmd_pub.publish(cmd)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = ImprovedStableDWA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
