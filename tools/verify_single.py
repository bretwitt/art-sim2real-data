#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from nav_msgs.msg import Odometry
from chrono_ros_interfaces.msg import DriverInputs as VehicleInput  # Assuming the message type for vehicle inputs
import matplotlib.pyplot as plt
import numpy as np
import os
from rclpy.serialization import deserialize_message
import tf_transformations  # Required to convert quaternion to Euler angles


class RosbagPlotter(Node):
    def __init__(self, bag_file):
        super().__init__('rosbag_plotter')
        self.bag_file = bag_file

        # Data containers for plotting
        self.odom_times = []
        self.odom_x = []
        self.odom_y = []
        self.heading = []  # Store the heading (yaw) angle here
        self.control_times = []
        self.steering = []
        self.throttle = []

        self.initial_yaw = None  # Store the initial yaw to rotate into the ENU frame
        self.initial_x = None    # Store the initial x position
        self.initial_y = None    # Store the initial y position

        self.plot_data()

    def plot_data(self):
        # Read the rosbag data
        self.read_rosbag()

        # Rotate the odometry data to the ENU frame and translate to start at (0,0)
        self.rotate_and_translate_odometry()

        # Plot the data after reading it
        self.create_plots()

    def read_rosbag(self):
        # Create rosbag reader and set up storage and converter options
        reader = SequentialReader()
        storage_options = StorageOptions(uri=self.bag_file, storage_id='sqlite3')  # Using SQLite3 as storage
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)
    
        # Get all topic types from the bag
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
    
        while reader.has_next():
            (topic, data, t) = reader.read_next()
    
            if topic == '/artcar_1/odometry/filtered':
                msg = deserialize_message(data, Odometry)
                self.odom_times.append(t)
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y

                if self.initial_x is None and self.initial_y is None:
                    # Store the initial x and y positions to translate the data
                    self.initial_x = x
                    self.initial_y = y

                self.odom_x.append(x)
                self.odom_y.append(y)
                
                # Extract yaw from the orientation quaternion
                orientation_q = msg.pose.pose.orientation
                quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                euler = tf_transformations.euler_from_quaternion(quaternion)
                yaw = euler[2]  # Yaw is the rotation around the z-axis

                if self.initial_yaw is None:
                    # Store the initial yaw to later rotate the data
                    self.initial_yaw = yaw

                self.heading.append(yaw)

            elif topic == '/artcar_1/control/vehicle_inputs':
                msg = deserialize_message(data, VehicleInput)
                self.control_times.append(t)
                self.steering.append(msg.steering)
                self.throttle.append(msg.throttle)

    def rotate_and_translate_odometry(self):
        if self.initial_yaw is None or self.initial_x is None or self.initial_y is None:
            return  # No odometry data processed, nothing to rotate or translate

        # Compute the rotation matrix to align the initial yaw to 0 (east direction)
        cos_theta = np.cos(-self.initial_yaw - np.pi/2)  # Negate the initial yaw for the correct rotation
        sin_theta = np.sin(-self.initial_yaw - np.pi/2)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        # Rotate and translate all (x, y) positions to start at (0, 0) and align with the ENU frame
        for i in range(len(self.odom_x)):
            # Translate the position relative to the initial position
            position = np.array([self.odom_x[i] - self.initial_x, self.odom_y[i] - self.initial_y])
            rotated_position = np.dot(rotation_matrix, position)
            self.odom_x[i], self.odom_y[i] = rotated_position

        # Adjust the heading (yaw) values to be relative to the new ENU frame
        self.heading = [yaw - self.initial_yaw for yaw in self.heading]

    def create_plots(self):
        # Convert timestamps to seconds for easier plotting
        odom_times_seconds = np.array(self.odom_times) * 1e-9
        control_times_seconds = np.array(self.control_times) * 1e-9
    
        # Plot Odometry data (x and y positions) in its own figure
        plt.figure(figsize=(3, 3))  # Set minimum height and width to 3 units
        plt.plot(self.odom_x, self.odom_y, label='Position (X-Y)')
        plt.title('/artcar_1/odometry/filtered (ENU Frame)')
        plt.xlabel('X Position [m]')
        plt.ylabel('Y Position [m]')
        plt.legend()
        plt.gca().set_aspect('equal', 'box')  # Ensures equal aspect ratio for X and Y
        plt.tight_layout()
        plt.show()

        # Create a new figure for the remaining subplots
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))  # Two subplots: Heading and Control Inputs
    
        # Plot Heading (Yaw)
        axs[0].plot(odom_times_seconds, self.heading, label='Heading (Yaw)', color='orange')
        axs[0].set_title('Vehicle Heading (Yaw) (ENU Frame)')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Yaw [rad]')
        axs[0].legend()
        axs[0].set_aspect('auto')
    
        # Plot Control inputs (steering and throttle)
        axs[1].plot(control_times_seconds, self.steering, label='Steering')
        axs[1].plot(control_times_seconds, self.throttle, label='Throttle')
        axs[1].set_title('/artcar_1/control/vehicle_inputs')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Control Inputs')
        axs[1].legend()
        axs[1].set_aspect('auto')
    
        # Adjust layout to prevent overlapping elements
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)

    # Path to your rosbag file
    #bag_file = '../trial_0D/playback_odometry_rosbag_1'
    bag_file = '../trial_3A/playback_odometry_rosbag_3'

    if not os.path.exists(bag_file):
        print(f"Bag file {bag_file} not found.")
        return

    rosbag_plotter = RosbagPlotter(bag_file)

    rclpy.shutdown()


if __name__ == '__main__':
    main()

