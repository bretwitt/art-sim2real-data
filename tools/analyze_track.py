#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import csv
import os
import math

class ErrorTracker(Node):
    def __init__(self):
        # Initialize the ROS 2 node
        super().__init__('error_tracker')

        # Subscribe to /artcar_2/odometry/filtered, /artcar_1/vehicle_traj, and /artcar_1/odometry/filtered
        self.odom_sub_2 = self.create_subscription(Odometry, '/artcar_2/odometry/filtered', self.odom_callback_2, 10)
        self.odom_sub_1 = self.create_subscription(Odometry, '/artcar_1/odometry/filtered', self.odom_callback_1, 10)
        self.path_sub = self.create_subscription(Path, '/artcar_1/vehicle_traj', self.path_callback, 10)

        # Initialize variables
        self.current_position_2 = None
        self.current_position_1 = None
        self.trajectory = None

        # Target trailing distance
        self.target_distance = 1.5

        # CSV file paths
        self.lateral_csv_file = 'lateral_error_data.csv'
        self.gap_csv_file = 'gap_error_data.csv'

        # Check if CSV files exist and add header if not
        if not os.path.exists(self.lateral_csv_file):
            with open(self.lateral_csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp (ms)', 'Lateral Error (m)'])

        if not os.path.exists(self.gap_csv_file):
            with open(self.gap_csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp (ms)', 'Gap Error (m)'])

        # Publishers for lateral error and gap error
        self.lateral_error_pub = self.create_publisher(PoseStamped, '/lateral_tracking_error', 10)
        self.gap_error_pub = self.create_publisher(PoseStamped, '/gap_error', 10)

    def odom_callback_2(self, odom_data):
        # Extract current position from the /artcar_2 Odometry message
        self.current_position_2 = odom_data.pose.pose.position

        # Compute lateral and gap errors if the trajectory and /artcar_1 position are available
        if self.trajectory is not None:
            self.compute_lateral_error()
        if self.current_position_1 is not None:
            self.compute_gap_error()

    def odom_callback_1(self, odom_data):
        # Extract current position from the /artcar_1 Odometry message
        self.current_position_1 = odom_data.pose.pose.position

        # Compute gap error if the /artcar_2 position is available
        if self.current_position_2 is not None:
            self.compute_gap_error()

    def path_callback(self, path_data):
        # Store the trajectory (Path message)
        self.trajectory = path_data.poses

        # Compute lateral error if the current position is available
        if self.current_position_2 is not None:
            self.compute_lateral_error()

    def compute_lateral_error(self):
        # Initialize variables
        closest_distance = float('inf')
        closest_point = None

        # Convert current position of artcar_2 to numpy array for easier manipulation
        current_pos = np.array([self.current_position_2.x, self.current_position_2.y])

        # Iterate through each pose in the trajectory
        for pose in self.trajectory:
            # Convert trajectory point to numpy array
            traj_pos = np.array([pose.pose.position.x, pose.pose.position.y])

            # Calculate Euclidean distance
            distance = np.linalg.norm(current_pos - traj_pos)

            # Find the closest point on the trajectory
            if distance < closest_distance:
                closest_distance = distance
                closest_point = traj_pos

        # Compute lateral error (perpendicular distance to the closest point)
        lateral_error = closest_distance

        # Log the lateral error and publish it
        self.get_logger().info(f'Lateral Tracking Error: {lateral_error}')

        # Publish the lateral error as a PoseStamped message
        error_msg = PoseStamped()
        error_msg.header.stamp = self.get_clock().now().to_msg()
        error_msg.pose.position.x = lateral_error
        self.lateral_error_pub.publish(error_msg)

        # Save lateral error to CSV
        self.save_lateral_error_to_csv(lateral_error=lateral_error)

    def compute_gap_error(self):
        # Convert positions of artcar_1 and artcar_2 to numpy arrays for easier manipulation
        pos_1 = np.array([self.current_position_1.x, self.current_position_1.y])
        pos_2 = np.array([self.current_position_2.x, self.current_position_2.y])

        # Compute the Euclidean distance between the two positions
        actual_gap = np.linalg.norm(pos_1 - pos_2)

        # Calculate the gap error based on the target distance
        gap_error = actual_gap - self.target_distance

        # Log the gap error and publish it
        self.get_logger().info(f'Gap Error (relative to target of {self.target_distance}m): {gap_error}')

        # Publish the gap error as a PoseStamped message
        gap_msg = PoseStamped()
        gap_msg.header.stamp = self.get_clock().now().to_msg()
        gap_msg.pose.position.x = gap_error
        self.gap_error_pub.publish(gap_msg)

        # Save gap error to CSV
        self.save_gap_error_to_csv(gap_error=gap_error)

    def save_lateral_error_to_csv(self, lateral_error):
        # Get current timestamp with milliseconds
        now = self.get_clock().now().to_msg()
        timestamp_ms = now.sec * 1000 + now.nanosec // 1000000  # Convert seconds and nanoseconds to milliseconds

        # Save to lateral error CSV
        with open(self.lateral_csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp_ms, lateral_error])

    def save_gap_error_to_csv(self, gap_error):
        # Get current timestamp with milliseconds
        now = self.get_clock().now().to_msg()
        timestamp_ms = now.sec * 1000 + now.nanosec // 1000000  # Convert seconds and nanoseconds to milliseconds

        # Save to gap error CSV
        with open(self.gap_csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp_ms, gap_error])


class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')

        # Publisher to /path
        self.path_pub = self.create_publisher(Path, '/path', 10)

        # Subscribe to /artcar_2/odometry/filtered to get the initial pose of the rover
        self.odom_sub = self.create_subscription(Odometry, '/artcar_1/odometry/filtered', self.odom_callback, 10)

        self.timer = self.create_timer(1.0, self.publish_path)  # Publish the path every second

        self.initial_position = None  # To store the initial position of the rover

    def odom_callback(self, odom_data):
        # Set the initial position only once
        if self.initial_position is None:
            self.initial_position = odom_data.pose.pose.position
            self.get_logger().info(f'Initial position set: x={self.initial_position.x}, y={self.initial_position.y}')

    def publish_path(self):
        if self.initial_position is None:
            self.get_logger().info('Waiting for initial position to be set...')
            return

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        mode = "CIRCLE"  # Choose mode ("STRAIGHT", "CIRCLE", or "SQUARE")

        if mode == "STRAIGHT":
            # Define a straight line path, say along the x-axis
            for i in range(50):
                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = 'map'
                pose.pose.position.x = self.initial_position.x + i * 0.25  # 0.5 meter intervals along x from the initial position
                pose.pose.position.y = self.initial_position.y  # Straight line along x, y = initial y
                pose.pose.position.z = 0.0  # Assuming a 2D plane
                path.poses.append(pose)

        elif mode == "CIRCLE":
            radius = 3.0
            num_points = 100  # Number of points to approximate the circle
            angle_increment = 2 * math.pi / num_points  # Angle step to complete a full circle

            center_x = self.initial_position.x + 0.5*radius  # Use initial position as the center
            center_y = self.initial_position.y

            for i in range(num_points):
                angle = i * angle_increment

                # Calculate position on the circle
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)

                # Calculate the heading as the tangent to the circle
                tangent_angle = angle + (math.pi / 2)  # Tangent is perpendicular to the radius

                # Manually calculate the quaternion from yaw (tangent_angle)
                q_z = math.sin(tangent_angle / 2)
                q_w = math.cos(tangent_angle / 2)

                # Create PoseStamped message
                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0  # Assuming a 2D plane

                # Set the orientation (heading) of the car
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = q_z
                pose.pose.orientation.w = q_w

                # Append the pose to the path
                path.poses.append(pose)

        elif mode == "SQUARE":
            side_length = 6.0  # Length of the square's side
            points_per_side = 45  # Number of points per side

            def create_pose(x, y, yaw):
                """Helper function to create a PoseStamped with a given position and yaw orientation."""
                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = 'map'

                # Set position
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0

                # Set orientation based on yaw
                q_z = math.sin(yaw / 2)
                q_w = math.cos(yaw / 2)
                pose.pose.orientation.z = q_z
                pose.pose.orientation.w = q_w

                return pose

            # First side: Moving right along the x-axis (yaw = 0)
            for i in range(points_per_side):
                x = self.initial_position.x + i * (side_length / points_per_side)
                y = self.initial_position.y
                yaw = 0.0  # Facing right
                path.poses.append(create_pose(x, y, yaw))

            # Second side: Moving up along the y-axis (yaw = π/2)
            for i in range(points_per_side):
                x = self.initial_position.x + side_length
                y = self.initial_position.y + i * (side_length / points_per_side)
                yaw = math.pi / 2  # Facing up
                path.poses.append(create_pose(x, y, yaw))

            # Third side: Moving left along the x-axis (yaw = π)
            for i in range(points_per_side):
                x = self.initial_position.x + side_length - i * (side_length / points_per_side)
                y = self.initial_position.y + side_length
                yaw = math.pi  # Facing left
                path.poses.append(create_pose(x, y, yaw))

            # Fourth side: Moving down along the y-axis (yaw = -π/2)
            for i in range(points_per_side):
                x = self.initial_position.x
                y = self.initial_position.y + side_length - i * (side_length / points_per_side)
                yaw = -math.pi / 2  # Facing down
                path.poses.append(create_pose(x, y, yaw))

        # Publish the path
        self.path_pub.publish(path)
        self.get_logger().info(f'{mode} Path Published')



def main(args=None):
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)

    # Create the nodes
    error_tracker = ErrorTracker()
    path_publisher = PathPublisher()

    # Create executor to spin both nodes
    executor = rclpy.executors.MultiThreadedExecutor()

    executor.add_node(error_tracker)
    executor.add_node(path_publisher)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # Shutdown the ROS 2 Python client library
    error_tracker.destroy_node()
    path_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()