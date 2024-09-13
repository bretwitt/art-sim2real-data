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
    def __init__(self, bag_file_1, bag_file_2=None):
        super().__init__('rosbag_plotter')
        self.bag_file_1 = bag_file_1
        self.bag_file_2 = bag_file_2

        # Data containers for the first bag file
        self.odom_times_1 = []
        self.odom_x_1 = []
        self.odom_y_1 = []
        self.heading_1 = []
        self.control_times_1 = []
        self.steering_1 = []
        self.throttle_1 = []

        # Data containers for the second bag file
        self.odom_times_2 = []
        self.odom_x_2 = []
        self.odom_y_2 = []
        self.heading_2 = []
        self.control_times_2 = []
        self.steering_2 = []
        self.throttle_2 = []

        # Store initial values for both bags
        self.initial_yaw_1 = None
        self.initial_x_1 = None
        self.initial_y_1 = None
        self.initial_yaw_2 = None
        self.initial_x_2 = None
        self.initial_y_2 = None

        # Store initial times for both bags
        self.initial_time_1 = None
        self.initial_time_2 = None

        self.plot_data()

    def plot_data(self):
        # Read and process the first rosbag data
        self.read_rosbag(self.bag_file_1, is_first_bag=True)

        # If a second bag file is provided, read and process that too
        if self.bag_file_2:
            self.read_rosbag(self.bag_file_2, is_first_bag=False)

        # Rotate and translate both datasets to the ENU frame
        self.rotate_and_translate_odometry(is_first_bag=True)
        if self.bag_file_2:
            self.rotate_and_translate_odometry(is_first_bag=False)

        # Plot the data after reading both files
        self.create_plots()

    def read_rosbag(self, bag_file, is_first_bag=True):
        # Create rosbag reader and set up storage and converter options
        reader = SequentialReader()
        storage_options = StorageOptions(uri=bag_file, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)

        while reader.has_next():
            (topic, data, t) = reader.read_next()

            if topic == '/artcar_1/odometry/filtered':
                msg = deserialize_message(data, Odometry)
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y

                if is_first_bag:
                    if self.initial_x_1 is None and self.initial_y_1 is None:
                        self.initial_x_1 = x
                        self.initial_y_1 = y
                        self.initial_time_1 = t  # Store initial time

                    self.odom_x_1.append(x)
                    self.odom_y_1.append(y)
                    self.odom_times_1.append(t - self.initial_time_1)  # Normalize time
                else:
                    if self.initial_x_2 is None and self.initial_y_2 is None:
                        self.initial_x_2 = x
                        self.initial_y_2 = y
                        self.initial_time_2 = t  # Store initial time
    
                    self.odom_x_2.append(x)
                    self.odom_y_2.append(y)
                    self.odom_times_2.append(t - self.initial_time_2)  # Normalize time

                # Extract yaw from the orientation quaternion
                orientation_q = msg.pose.pose.orientation
                quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                euler = tf_transformations.euler_from_quaternion(quaternion)
                yaw = euler[2]  # Yaw is the rotation around the z-axis

                # Store the initial yaw if processing the first message
                if is_first_bag:
                    if self.initial_yaw_1 is None:
                        self.initial_yaw_1 = yaw
                    self.heading_1.append(yaw)
                else:
                    if self.initial_yaw_2 is None:
                        self.initial_yaw_2 = yaw
                    self.heading_2.append(yaw)

            elif topic == '/artcar_1/control/vehicle_inputs':
                msg = deserialize_message(data, VehicleInput)

                if is_first_bag:
                    self.control_times_1.append(t - self.initial_time_1)  # Normalize time
                    self.steering_1.append(msg.steering)
                    self.throttle_1.append(msg.throttle)
                else:
                    if(self.initial_time_2 == None):
                        self.initial_time_2 = t

                    self.control_times_2.append(t - self.initial_time_2)  # Normalize time
                    self.steering_2.append(msg.steering)
                    self.throttle_2.append(msg.throttle)

    def rotate_and_translate_odometry(self, is_first_bag=True):
        if is_first_bag:
            initial_x, initial_y, initial_yaw = self.initial_x_1, self.initial_y_1, self.initial_yaw_1
            odom_x, odom_y, heading = self.odom_x_1, self.odom_y_1, self.heading_1
        else:
            initial_x, initial_y, initial_yaw = self.initial_x_2, self.initial_y_2, self.initial_yaw_2
            odom_x, odom_y, heading = self.odom_x_2, self.odom_y_2, self.heading_2

        if initial_yaw is None or initial_x is None or initial_y is None:
            return  # No odometry data processed, nothing to rotate or translate

        # Compute the rotation matrix to align the initial yaw to 0 (east direction)
        cos_theta = np.cos(-initial_yaw)
        sin_theta = np.sin(-initial_yaw)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        # Rotate and translate all (x, y) positions to start at (0, 0) and align with the ENU frame
        for i in range(len(odom_x)):
            position = np.array([odom_x[i] - initial_x, odom_y[i] - initial_y])
            rotated_position = np.dot(rotation_matrix, position)
            odom_x[i], odom_y[i] = rotated_position

        # Adjust the heading (yaw) values to be relative to the new ENU frame
        heading[:] = [yaw - initial_yaw for yaw in heading]

    def create_plots(self):
        # Convert timestamps to seconds for easier plotting
        odom_times_seconds_1 = np.array(self.odom_times_1) * 1e-9
        control_times_seconds_1 = np.array(self.control_times_1) * 1e-9
        if self.bag_file_2:
            odom_times_seconds_2 = np.array(self.odom_times_2) * 1e-9
            control_times_seconds_2 = np.array(self.control_times_2) * 1e-9

        # Plot Odometry data (x and y positions) in its own figure
        plt.figure(figsize=(3, 3))  # Set minimum height and width to 3 units
        plt.plot(self.odom_x_1, self.odom_y_1, label='Bag 1 Position (X-Y)')
        if self.bag_file_2:
            plt.plot(self.odom_x_2, self.odom_y_2, label='Bag 2 Position (X-Y)', linestyle='--')
        plt.title('Odometry (ENU Frame)')
        plt.xlabel('X Position [m]')
        plt.ylabel('Y Position [m]')
        plt.legend()
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.show()

        # Create a new figure for the remaining subplots
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))

        # Plot Heading (Yaw)
        axs[0].plot(odom_times_seconds_1, self.heading_1, label='Bag 1 Heading (Yaw)', color='orange')
        if self.bag_file_2:
            axs[0].plot(odom_times_seconds_2, self.heading_2, label='Bag 2 Heading (Yaw)', color='blue', linestyle='--')
        axs[0].set_title('Vehicle Heading (Yaw) (ENU Frame)')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Yaw [rad]')
        axs[0].legend()
        axs[0].set_aspect('auto')

        # Plot Control inputs (steering and throttle)
        axs[1].plot(control_times_seconds_1, self.steering_1, label='Bag 1 Steering')
        axs[1].plot(control_times_seconds_1, self.throttle_1, label='Bag 1 Throttle')
        if self.bag_file_2:
            axs[1].plot(control_times_seconds_2, self.steering_2, label='Bag 2 Steering', linestyle='--')
            axs[1].plot(control_times_seconds_2, self.throttle_2, label='Bag 2 Throttle', linestyle='--')
        axs[1].set_title('Control Inputs')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Control Inputs')
        axs[1].legend()
        axs[1].set_aspect('auto')

        # Adjust layout to prevent overlapping elements
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)

    # Paths to your rosbag files
    bag_file_1 = '../trial_0C/record_rosbag'
    bag_file_2 = '../trial_0C/playback_odometry_rosbag'  # Example second bag file path
    
    if not os.path.exists(bag_file_1):
        print(f"Bag file {bag_file_1} not found.")
        return
    
    if not os.path.exists(bag_file_2):
        print(f"Bag file {bag_file_2} not found.")
        return

    rosbag_plotter = RosbagPlotter(bag_file_1, bag_file_2)

    rclpy.shutdown()


if __name__ == '__main__':
    main()

