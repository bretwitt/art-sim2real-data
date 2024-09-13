#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

def plot_errors(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert timestamp to a readable format (optional, depending on how you stored it)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    # Plot lateral error
    plt.figure(figsize=(10, 5))
    plt.plot(df['Timestamp'], df['Lateral Error (m)'], label='Lateral Error', color='blue', marker='o')
    plt.xlabel('Timestamp')
    plt.ylabel('Lateral Error (m)')
    plt.title('Lateral Tracking Error Over Time')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the first plot
    plt.show()

    # Plot gap error
    plt.figure(figsize=(10, 5))
    plt.plot(df['Timestamp'], df['Gap Error (m)'], label='Gap Error', color='red', marker='o')
    plt.xlabel('Timestamp')
    plt.ylabel('Gap Error (m)')
    plt.title('Gap Error Over Time (relative to 1.5m target)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the second plot
    plt.show()

if __name__ == '__main__':
    # Path to the CSV file
    csv_file = 'error_data.csv'

    # Call the plotting function
    plot_errors(csv_file)

