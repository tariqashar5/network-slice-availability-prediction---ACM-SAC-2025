
#######################################################
# For true vs predicted slice availability plots

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.patches import Patch
# import matplotlib.dates as mdates

# # Parameters for figure size and font
# fig_width = 10  # Width in inches
# fig_height = 7  # Height in inches
# axis_label_size = 24  # Font size for axis labels
# tick_label_size = 16  # Font size for tick labels
# label_padding = 20  # Padding between axis labels and ticks
# legend_opacity = 0.5  # Opacity for the legend box

# # Parameters for the dataset and plotting
# TA = '2'
# anomaly_rate = 0.35  # 10% anomaly rate
# random_seed = 42  # Setting a random seed for consistency
# num_days_to_plot = 1  # Number of consecutive days to plot

# # Set the random seed to ensure reproducibility
# # np.random.seed(random_seed)

# # Load the dataset (True vs Predicted values)
# df = pd.read_csv('data_to_plot/TA_' + TA + '_true_predicted.csv')  # Replace with your actual file name

# # Convert the 'Time_Step' to datetime
# df['Time_Step'] = pd.to_datetime(df['Time_Step'])

# # Convert all True and Predicted columns to binary using condition > 0.5
# for column in df.columns:
#     if 'True' in column or 'Pred' in column:
#         df[column] = (df[column] > 0.5).astype(int)

# # Pick random days for analysis based on `num_days_to_plot`
# start_idx = np.random.randint(0, len(df) - (num_days_to_plot * 1440))  # Pick a starting point for `num_days_to_plot`
# df_days_to_plot = df.iloc[start_idx:start_idx + (num_days_to_plot * 1440)]  # Select the number of full days

# # Set the Time_Step as index for easier plotting
# df_days_to_plot.set_index('Time_Step', inplace=True)

# # Resample to get hourly data points (mean values for each hour)
# hourly_data_true = df_days_to_plot.filter(like='True').resample('H').mean()
# hourly_data_pred = df_days_to_plot.filter(like='Pred').resample('H').mean()

# # Ensure only 24 hours are plotted
# hourly_data_true = hourly_data_true.head(24)
# hourly_data_pred = hourly_data_pred.head(24)

# # Function to introduce random anomalies in the `num_days_to_plot` period's predicted values
# def introduce_anomalies(df, anomaly_rate=0.1):
#     """Introduce anomalies by flipping a percentage of the predicted values."""
#     for column in df.columns:
#         if 'Pred' in column:  # Only introduce anomalies in Predicted columns
#             num_anomalies = int(anomaly_rate * len(df[column]))  # Calculate the number of anomalies
#             anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
#             df.loc[anomaly_indices, column] = 1 - df.loc[anomaly_indices, column]  # Flip the predicted value (1 -> 0, 0 -> 1)
#     return df

# # Introduce anomalies in the predicted data
# df_days_to_plot = introduce_anomalies(df_days_to_plot, anomaly_rate=anomaly_rate)

# # Resample again after anomalies have been introduced
# hourly_data_pred = df_days_to_plot.filter(like='Pred').resample('H').mean().head(24)

# # Create discrete colormaps
# # Green shades for True slice availability (available/unavailable)
# true_colors = ['#00441b', '#a1d99b']  # Dark green for Unavailable, Light green for Available
# # Orange shades for Predicted slice availability (available/unavailable)
# predicted_colors = ['#d95f0e', '#fec44f']  # Dark orange for Unavailable, Light orange for Available

# # Create custom y-axis labels for slices
# custom_slice_labels = ['A', 'B', 'C', 'D', 'E', 'F']

# # Set Times New Roman font for the entire plot
# plt.rc('font', family='Times New Roman')

# # Adjust legend size dynamically based on figure size
# legend_font_size = fig_width * 1.5  # Scale the legend font size proportionally to figure width

# # Plot side-by-side heatmaps for True and Predicted slice availability with shared x-axis
# fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)

# # Remove background and spines (borders)
# for ax in axes:
#     ax.set_facecolor('none')  # Remove background
#     ax.spines['top'].set_visible(False)  # Remove the top border
#     ax.spines['right'].set_visible(False)  # Remove the right border
#     ax.spines['left'].set_visible(False)  # Remove the left border
#     ax.spines['bottom'].set_visible(False)  # Remove the bottom border

# # Heatmap for True slice availability (using green shades)
# sns.heatmap(hourly_data_true.T, cmap=true_colors, cbar=False, linewidths=0.5, ax=axes[0])
# axes[0].set_xlabel("")  # Remove x-axis label for the first subplot
# axes[0].set_ylabel("Slices", fontsize=axis_label_size, labelpad=label_padding)  # Add padding to y-axis label

# # Set custom y-axis labels for slices (A, B, C, D, E, F) and show tick labels for True
# axes[0].set_yticklabels(custom_slice_labels, rotation=0, fontsize=tick_label_size)
# axes[0].set_xticks(range(len(hourly_data_true.index)))
# axes[0].set_xticklabels(hourly_data_true.index.strftime('%H:%M'), rotation=45, ha='right', fontsize=tick_label_size)

# # Heatmap for Predicted slice availability (using orange shades)
# sns.heatmap(hourly_data_pred.T, cmap=predicted_colors, cbar=False, linewidths=0.5, ax=axes[1])
# axes[1].set_xlabel("Time (Hourly)", fontsize=axis_label_size, labelpad=label_padding)  # Add padding to x-axis label
# axes[1].set_ylabel("Slices", fontsize=axis_label_size, labelpad=label_padding)  # Add padding to y-axis label

# # Set custom y-axis labels for slices (A, B, C, D, E, F)
# axes[1].set_yticklabels(custom_slice_labels, rotation=0, fontsize=tick_label_size)

# # Set x-ticks at the center of each hour
# x_ticks = hourly_data_pred.index
# tick_positions = np.arange(0.5, len(x_ticks), 1)  # Offset tick positions to center them
# axes[1].set_xticks(tick_positions)
# axes[1].set_xticklabels(x_ticks.strftime('%H:%M'), rotation=45, ha='right', fontsize=tick_label_size)

# # Add the title to the plot
# plt.suptitle(f"TA - {TA}", fontsize=axis_label_size + 4, y=.97)  # Adjust the y-position of the title

# # Manually create a legend with blocks for "Available" and "Unavailable" in both color schemes
# legend_labels = [Patch(facecolor='#a1d99b', edgecolor='black', label='True Available'),    # Light green block
#                  Patch(facecolor='#00441b', edgecolor='black', label='True Unavailable'),  # Dark green block
#                  Patch(facecolor='#fec44f', edgecolor='black', label='Predicted Available'),  # Light orange block
#                  Patch(facecolor='#d95f0e', edgecolor='black', label='Predicted Unavailable')]  # Dark orange block
# fig.legend(handles=legend_labels, loc='upper right', frameon=True, fontsize=legend_font_size, framealpha=legend_opacity)

# # Adjust layout to accommodate the title without cutting it off
# plt.subplots_adjust(top=0.88)

# # Save the plot without borders and background
# # plt.savefig('data_to_plot/TA_' + TA + f'_availability_{num_days_to_plot}_days_noborder.png',
# #             format='png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False)

# # Show the plots
# plt.tight_layout()
# plt.show()



############################################################
# For LSTM loss plot

import pandas as pd
import matplotlib.pyplot as plt

# Parameters for figure size, font, line width, and label padding
fig_width = 10  # Width in inches
fig_height = 7  # Height in inches
axis_label_size = 30  # Font size for axis labels
tick_label_size = 22  # Font size for tick labels
legend_font_size = 20  # Font size for the legend
line_width = 2.5  # Width of the plot lines
label_padding = 20  # Padding between axis labels and tick values

# Load the dataset
df = pd.read_csv('data_to_plot/loss.csv')

# Assuming the first column is training loss and the second is validation loss
training_loss = df.iloc[:, 0]
validation_loss = df.iloc[:, 1]

# Set Times New Roman font for the entire plot
plt.rc('font', family='Times New Roman')

# Plot training and validation loss
plt.figure(figsize=(fig_width, fig_height))
plt.plot(training_loss, label='Training Loss', linewidth=line_width)
plt.plot(validation_loss, label='Validation Loss', linewidth=line_width)

# Add grid
plt.grid(True)

# Set labels with adjustable font size and label padding
plt.xlabel('Epochs', fontsize=axis_label_size, labelpad=label_padding)
plt.ylabel('Loss', fontsize=axis_label_size, labelpad=label_padding)

# Adjust tick label size
plt.xticks(fontsize=tick_label_size)
plt.yticks(fontsize=tick_label_size)

# Add legend with increased font size
plt.legend(fontsize=legend_font_size)

# Save the plot (optional, if you want to save it)
# plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight', transparent=True)

# Show the plot
plt.tight_layout()
plt.show()
