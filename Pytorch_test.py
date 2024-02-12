#!/usr/bin/env python
# coding: utf-8

# In[19]:

from tqdm import tqdm
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


class Parcentile975(nn.Module):
    def __init__(self):
        super(Parcentile975, self).__init__()
    def forward(self, y_true, y_pred):
        error = torch.abs(y_true - y_pred)
        return torch.parcentile(error, 97.5)


# In[6]:


class CustomModel(nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 320)
        self.fc8 = nn.Linear(320, 320)
        self.fc9 = nn.Linear(320, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 256)
        self.fc14 = nn.Linear(256, 128)
        self.fc15 = nn.Linear(128, 64)
        self.fc16 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.fc1(x))
        x = torch.nn.functional.gelu(self.fc2(x))
        x = torch.nn.functional.gelu(self.fc3(x))
        x = torch.nn.functional.gelu(self.fc4(x))
        x = torch.nn.functional.gelu(self.fc5(x))
        x = torch.nn.functional.gelu(self.fc6(x))
        x = torch.nn.functional.gelu(self.fc7(x))
        x = torch.nn.functional.gelu(self.fc8(x))
        x = torch.nn.functional.gelu(self.fc9(x))
        x = torch.nn.functional.gelu(self.fc10(x))
        x = torch.nn.functional.gelu(self.fc11(x))
        x = torch.nn.functional.gelu(self.fc12(x))
        x = torch.nn.functional.gelu(self.fc13(x))
        x = torch.nn.functional.gelu(self.fc14(x))
        x = torch.nn.functional.gelu(self.fc15(x))
        x = torch.nn.functional.gelu(self.fc16(x))
        return x


# In[7]:


def load_and_preprocess_data_for_separate_datasets(f_dir_train_csv, param_col, result_str):
    df_train = pd.read_csv(f_dir_train_csv, delimiter=';')
    df_train = df_train.dropna()

    y_train = df_train['Mises'].values
    X_train = df_train[param_col].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=95)

    return X_train, X_val, y_train, y_val


# In[29]:


def train_model(model, train_loader, val_loader, dir_save_model, epochs, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0001, lr=0.0001)

    val_loss_min = float('inf')  # Initialize to positive infinity
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs.squeeze(), targets.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                outputs = model(inputs.to(device))
                loss = criterion(outputs.squeeze(), targets.to(device))
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        end_time = time.time()  # Record the end time of the epoch
        epoch_time = end_time - start_time  # Calculate the duration of the epoch

        print(f'Epoch: {epoch+1}/{epochs} - Training Loss: {train_loss} - Validation Loss: {val_loss} - Epoch Time: {epoch_time:.2f} seconds')

        if val_loss <= val_loss_min:
            print(f'Validation loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), os.path.join(dir_save_model, f'model_epoch{epoch+1}.pt'))
            val_loss_min = val_loss

    return train_losses, val_losses


# In[30]:


def evaluate_and_plot(model, train_losses, val_loader, device, dir_save_plot_1, dir_save_plot_2,  dir_save_plot_3):
    model.to(device)
    criterion = nn.MSELoss()

    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in tqdm(val_loader):
            outputs = model(inputs.to(device))
            loss = criterion(outputs.squeeze(), targets.to(device))
            val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print("Test Loss: ", val_loss)

        y_true = []
        y_pred = []
        for inputs, targets in tqdm(val_loader):
            outputs = model(inputs.to(device))
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(outputs.cpu().numpy().flatten())

        error = np.array(y_pred) - np.array(y_true)
        plt.hist(error, bins=400)
        plt.xlabel("Error")
        plt.ylabel("Number of samples")
        plt.xlim((-30, 30))
        plt.title("Distribution of test error")
        plt.savefig(os.path.join(dir_save_plot_1), dpi=600, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Predicted vs. Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted Values vs. actual')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(dir_save_plot_2), dpi=600, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time MSE')
        plt.savefig(os.path.join(dir_save_plot_3), dpi=600, bbox_inches='tight')
        plt.close()
    


# In[ ]:


if __name__ == "__main__":
    param_col = ['[degrees]basis_angle_alpha', '[degrees]tangent_angle_r1_2',
                 '[mm]ae_blade', '[mm]base_width_blade', '[degrees]beta1',
                 '[mm]a1', '[mm]a2', '[mm]b1', '[mm]b2', '[mm]l1', '[mm]l2',
                 '[mm]t1', '[mm]t2', '[mm]r_blade1_3', '[mm]r_blade1_4',
                 '[mm]r_blade2_3', '[mm]r_blade2_4', '[mm]r_disk1_1',
                 '[mm]r_disk1_2', '[mm]r_disk2_1', '[mm]r_disk2_2',
                 '[degrees]gamma_blade2', '[degrees]gamma_blade3', 'T',
                 'E_x', 'zomega', 'Points:0', 'Points:1', 'Points:2', 'rdensity']

    epochs = 10
    result_str = 'Mises'

    dir_save_model = r"/home/h3/tais843f/validation_result_firtree/"

    dir_save_plot_1 = r"/home/h3/tais843f/validation_result_firtree/error_distribution_plot"
    dir_save_plot_2 = r"/home/h3/tais843f/validation_result_firtree/predicted_vs_true_plot"
    dir_save_plot_3 = r"/home/h3/tais843f/validation_result_firtree/training_vs_test_plot"
    
    f_dir_train_csv = r"/home/h3/tais843f/Training_data/training_100mcs_1grad_firtree_23variable.csv"

    X_train, X_val, y_train, y_val = load_and_preprocess_data_for_separate_datasets(f_dir_train_csv,
                                                                                      param_col,
                                                                                      result_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    model = CustomModel(input_size=len(param_col))
    train_losses, val_losses = train_model(model, train_loader, val_loader, dir_save_model, epochs, device)
    evaluate_and_plot(model, train_losses, val_loader, device, dir_save_plot_1, dir_save_plot_2,  dir_save_plot_3)


# In[ ]:




