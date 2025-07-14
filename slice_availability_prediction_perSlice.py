# ================================
# Hyperparameters (modifiable)
# ================================
learning_rate = 0.00005  # Modify this value to experiment with different learning rates
lags = 12  # Number of time lags for feature generation
dropout = 0.4
train_test_split_ratio = 0.8  # Ratio for train/test split
batch_size = 128  # Batch size for training
epochs = 30  # Number of training epochs
hidden_units = 64  # Number of hidden units in the model
model_type = 'lstm'  # Choose 'lstm', 'gru', or 'saes'
binary_classification = True  # Set to True for binary output predictions
input_file = 'TA_10.csv'  # Path to the dataset file
slice_columns = [f'S-NSSAI_{i}' for i in range(1, 7)]  # Columns for each slice in the TA file

# ================================
# Import necessary libraries
# ================================
import os
import numpy as np
import pandas as pd
import sys
import warnings
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# Ensure directory exists
output_folder = 'model_results/TA_10'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ================================
# Data Processing
# ================================
def process_data(df, attr, lags, train_test_split_ratio):
    """Process data for a given slice and split into train/test sets."""
    
    # Normalize the prediction column values
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df[attr].values.reshape(-1, 1))
    normalized_data = scaler.transform(df[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # Split the data into training and testing sets based on time
    split_index = int(len(normalized_data) * train_test_split_ratio)
    train_data = normalized_data[:split_index]
    test_data = normalized_data[split_index:]

    # Create lagged datasets for training and testing
    train, test = [], []
    for i in range(lags, len(train_data)):
        train.append(train_data[i - lags: i + 1])
    for i in range(lags, len(test_data)):
        test.append(test_data[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    # Split into features (X) and labels (y)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    # Convert labels to binary if binary classification is required
    if binary_classification:
        y_train = (y_train > 0.5).astype(int)
        y_test = (y_test > 0.5).astype(int)

    return X_train, y_train, X_test, y_test, scaler, len(train_data)

# ================================
# Model Definitions (LSTM)
# ================================
def get_lstm(units):
    """Build LSTM Model."""
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(BatchNormalization())  # Add batch normalization
    model.add(LSTM(units[2]))
    model.add(BatchNormalization())  # Add batch normalization
    model.add(Dropout(dropout))
    model.add(Dense(units[3], activation='sigmoid' if binary_classification else 'linear',
                    kernel_regularizer=None))  # Add L2 regularization if needed
    return model

# ================================
# Training Functions
# ================================
def train_model(model, X_train, y_train, X_test, y_test, name, config):
    """Train the neural network model, save training/validation metrics, and plots."""
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy" if binary_classification else "mse", 
                  optimizer=optimizer, 
                  metrics=['accuracy' if binary_classification else 'mape'])
    
    # Train the model and store history
    history = model.fit(X_train, y_train, 
                        batch_size=config["batch"], 
                        epochs=config["epochs"], 
                        validation_split=0.2)
    
    # Save the model
    # model.save(f'{output_folder}/{name}.h5')
    
    # Save training history as CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'{output_folder}/{name}_training_history.csv', index=False)
    
    # Plot training & validation loss and accuracy
    plot_training_history(history.history, name)
    
    # Evaluate on the test set and plot true vs predicted values
    y_pred = model.predict(X_test)
    
    return y_pred, y_test  # Return the predictions and true values

# ================================
# Plotting Functions
# ================================
def plot_training_history(history, name):
    """Plot training and validation loss and accuracy, and save to file."""
    
    # Loss Plot
    plt.figure()
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{name} - Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_folder}/{name}_loss_plot.png')
    plt.close()
    
    # Accuracy Plot
    if 'accuracy' in history:
        plt.figure()
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{name} - Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_folder}/{name}_accuracy_plot.png')
        plt.close()

def save_true_vs_pred_csv(slice_name, y_true, y_pred):
    """Save true vs predicted values to a CSV file for comparison."""
    df_true_vs_pred = pd.DataFrame({
        'True': y_true.flatten(),
        'Predicted': y_pred.flatten()
    })
    df_true_vs_pred.to_csv(f'{output_folder}/{slice_name}_true_vs_pred.csv', index=False)
    print(f"True vs Predicted values saved to {slice_name}_true_vs_pred.csv")

# ================================
# Main Function (Run LSTM for Each Slice and Save Predictions)
# ================================
def main():
    config = {"batch": batch_size, "epochs": epochs}
    
    # Read the input TA file
    df = pd.read_csv(input_file)
    
    # Prepare a DataFrame to store the predicted values
    predicted_values = pd.DataFrame()
    predicted_values['Time_Step'] = df['Time_Step']  # Assuming you have a 'Time_Step' column
    
    # Loop through each slice column and train the LSTM for each slice
    for slice_column in slice_columns:
        print(f"Training model for {slice_column}...")
        
        # Process data for the current slice
        X_train, y_train, X_test, y_test, scaler, train_data_len = process_data(df, slice_column, lags, train_test_split_ratio)

        # Reshape input data for the model
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Initialize and build the LSTM model
        model = get_lstm([lags, hidden_units, hidden_units, 1])
        
        # Train and get predictions
        y_pred, y_true = train_model(model, X_train, y_train, X_test, y_test, f'lstm_{slice_column}', config)
        
        # Inverse scale the predictions and true values if necessary
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_true_rescaled = scaler.inverse_transform(y_true.reshape(-1, 1))
        
        # Save the true vs predicted values for comparison
        save_true_vs_pred_csv(slice_column, y_true_rescaled, y_pred_rescaled)

        # Predict the next 24 hours (future slice availability)
        future_preds = []
        
        # Ensure we are using exactly 'lags' values for prediction
        input_sequence = X_test[-1]  # Last sequence from the test set
        input_sequence = input_sequence.reshape((1, lags, 1))  # Reshape the input for LSTM

        for i in range(1440):  # 1440 minutes for 24 hours
            next_pred = model.predict(input_sequence)
            future_preds.append(next_pred.flatten())  # Store the prediction
            
            # Update input sequence for the next prediction (rolling window)
            input_sequence = np.append(input_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

        # Inverse scale future predictions
        future_preds_rescaled = scaler.inverse_transform(np.array(future_preds))

        # Append future predicted values for each slice
        predicted_values[f'Future_Predicted_{slice_column}'] = np.concatenate([np.full(len(df) - len(future_preds_rescaled), np.nan), future_preds_rescaled.flatten()])

    # Save the predicted values for future availability to a CSV file
    predicted_values.to_csv(f'{output_folder}/TA_1_predicted_24hrs.csv', index=False)
    print("Future predicted slice availability saved to TA_1_predicted_24hrs.csv")

if __name__ == '__main__':
    main()

