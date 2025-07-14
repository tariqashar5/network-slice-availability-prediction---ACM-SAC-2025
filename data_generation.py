import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# Define the parameters for the dataset
num_TAs = 10  # Number of TAs
num_slices = 6  # Number of slices (S-NSSAI_1, S-NSSAI_2, etc.)
start_time = "2024-09-22 00:00"  # Starting time
days = 14  # Number of days to generate data
time_interval_minutes = 1  # Define the timestep interval in minutes

# Automatically calculate the end time based on the number of days
start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
end_dt = start_dt + timedelta(days=days)

# Define a dynamic slice trend generation function based on the number of slices
def get_slice_trend(hour, weekday, num_slices):
    slice_trend = [0] * num_slices  # Initialize availability list with 0s

    # Example trends based on time of day and weekday/weekend
    if weekday:  # Weekdays
        if 6 <= hour < 12:
            slice_trend[:num_slices//3] = [1] * (num_slices//3)  # Morning slice trend for 1/3rd of slices
        elif 12 <= hour < 18:
            slice_trend[:2*num_slices//3] = [1] * (2*num_slices//3)  # Afternoon for 2/3rds of slices
        elif 18 <= hour < 22:
            slice_trend[:] = [1] * num_slices  # Evening for all slices
        else:
            slice_trend[-(num_slices//2):] = [1] * (num_slices//2)  # Night for last half of slices
    else:  # Weekends
        if 8 <= hour < 12:
            slice_trend[::2] = [1] * (num_slices//2)  # Morning weekend for alternating slices
        elif 12 <= hour < 18:
            slice_trend[:] = [1] * num_slices  # Afternoon weekend for all slices
        elif 18 <= hour < 22:
            slice_trend[:num_slices//2] = [1] * (num_slices//2)  # Evening weekend for first half of slices
        else:
            slice_trend[-1] = 1  # Late-night weekend for only the last slice

    return slice_trend

# Generate the time intervals
time_range = pd.date_range(start=start_dt, end=end_dt, freq=f'{time_interval_minutes}T')

# Loop through each TA and generate separate files
for ta in range(1, num_TAs + 1):
    data = []
    for time_step in time_range:
        weekday = time_step.weekday() < 5  # True if it's a weekday, False if weekend
        hour = time_step.hour
        
        # Determine the slice availability trend for this TA at the current time step
        slice_availability = get_slice_trend(hour, weekday, num_slices)
        
        # Add the TA, time step, and slice availability for each time step
        data.append([f'TA_{ta}', time_step] + slice_availability)
    
    # Convert to a DataFrame with dynamic column names for slices
    slice_columns = [f'S-NSSAI_{i+1}' for i in range(num_slices)]
    df = pd.DataFrame(data, columns=['TA', 'Time_Step'] + slice_columns)
    
    # Save the generated dataset for this TA to a separate CSV file
    output_file = f'TA_{ta}.csv'
    df.to_csv(output_file, index=False)
    print(f'Generated file for TA_{ta}: {output_file}')

# Display a message when all files are generated
print("All TA files have been generated.")
