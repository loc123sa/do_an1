import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import numpy as np
import math


class StableDWA(Node):

    def __init__(self):

        super().__init__('stable_dwa')

       
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

        # DATA
      

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

    
        # TIMER
        

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            'FINAL STABLE DWA STARTED'
        )

   
    # SCAN CALLBACK
   

    def scan_callback(self, msg):

        self.scan_data = np.array(msg.ranges)

   
    # ODOM CALLBACK
   

    def odom_callback(self, msg):

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        self.yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

    
    # TRAJECTORY SIMULATION

    def simulate_trajectory(self, v, w):

        x = self.x
        y = self.y
        yaw = self.yaw

        dt = 0.1

        for _ in range(
            int(self.predict_time / dt)
        ):

            x += v * math.cos(yaw) * dt

            y += v * math.sin(yaw) * dt

            yaw += w * dt

        return x, y, yaw

    # GOAL COST


    def goal_cost(self, x, y):

        return math.sqrt(
            (self.goal_x - x) ** 2 +
            (self.goal_y - y) ** 2
        )

    # CONTROL LOOP
  

    def control_loop(self):

        if self.scan_data is None:
            return

        cmd = Twist()

     
        # WIDE FRONT DETECTION
        # Robot:
        # front = beginning of scan array
       

        front_center = self.scan_data[0:25]

        front_left = self.scan_data[25:80]

        front_right = self.scan_data[-80:-25]

        left = self.scan_data[80:140]

        right = self.scan_data[-140:-80]

       
        # REMOVE INVALID VALUES
       

        front_center = front_center[
            np.isfinite(front_center)
        ]

        front_left = front_left[
            np.isfinite(front_left)
        ]

        front_right = front_right[
            np.isfinite(front_right)
        ]

        left = left[np.isfinite(left)]

        right = right[np.isfinite(right)]

        if len(front_center) == 0:
            return

        front_min = np.min(front_center)

        front_left_min = np.min(front_left) \
            if len(front_left) > 0 else 999

        front_right_min = np.min(front_right) \
            if len(front_right) > 0 else 999

        left_mean = np.mean(left) \
            if len(left) > 0 else 0.0

        right_mean = np.mean(right) \
            if len(right) > 0 else 0.0

      
        # EMERGENCY OBSTACLE AVOIDANCE
        

        if (
            front_min < 0.45 or
            front_left_min < 0.35 or
            front_right_min < 0.35
        ):

            cmd.linear.x = 0.0

            # choose free direction
            if left_mean > right_mean:

                cmd.angular.z = 0.6

            else:

                cmd.angular.z = -0.6

            self.cmd_pub.publish(cmd)

            return

       
        # DWA SEARCH
       

        best_cost = 999999

        best_v = 0.12

        best_w = 0.0

        for v in np.arange(
            0.08,
            self.max_speed,
            self.v_resolution
        ):

            for w in np.arange(
                -self.max_yawrate,
                self.max_yawrate,
                self.w_resolution
            ):

                tx, ty, tyaw = \
                    self.simulate_trajectory(v, w)

                # distance to goal
                g_cost = self.goal_cost(
                    tx,
                    ty
                )

                # prefer straight
                turn_cost = abs(w)

                # prefer faster forward
                speed_cost = (
                    self.max_speed - v
                )

                total_cost = (
                    1.5 * g_cost +
                    0.4 * turn_cost +
                    0.2 * speed_cost
                )

                if total_cost < best_cost:

                    best_cost = total_cost

                    best_v = v

                    best_w = w

    
        # PUBLISH CMD
        

        cmd.linear.x = float(best_v)

        cmd.angular.z = float(best_w)

        self.cmd_pub.publish(cmd)


def main(args=None):

    rclpy.init(args=args)

    node = StableDWA()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()