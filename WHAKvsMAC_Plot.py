import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
folder_path = 'Figures1/Earth/Escape_on/'
file_whak = folder_path + 'Area_WHAK.csv'
file_mac = folder_path + 'Area_MAC.csv'

# Load data
whak_data = pd.read_csv(file_whak)
mac_data = pd.read_csv(file_mac)

# Extract time column (assumed to be the same for both)
time = whak_data['time']

# Case labels
test_case_labels = ["0.1% Earth Ocean", "1% Earth Ocean", "10% Earth Ocean", "100% Earth Ocean"]
num_cases = len(test_case_labels)

# Generate colors: different shades for each case
colors_mac = plt.cm.Blues(np.linspace(0.3, 0.9, num_cases))   # Blue shades for MAC
colors_whak = plt.cm.Oranges(np.linspace(0.3, 0.9, num_cases))  # Orange shades for WHAK

# Set up the plot
plt.figure(figsize=(10, 6))


# Plot MAC model in blue shades
for i in range(num_cases):
    plt.plot(time, mac_data[f'temp_case_{i+1}'], color=colors_mac[i], linestyle='-',alpha=0.9,
             label=f'MAC - {test_case_labels[i]}')

# Plot WHAK model in orange shades
for i in range(num_cases):
    plt.plot(time, whak_data[f'temp_case_{i+1}'], color=colors_whak[i], linestyle='--',alpha=0.8,
             label=f'WHAK - {test_case_labels[i]}')


# Labels and legend
plt.xlabel('Time (Gyr)')
plt.ylabel('Surface Temperature (K)')
#plt.title('Comparison of WHAK and MAC Model Runs')
plt.legend()
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.grid()

save_path = 'Figures1/Earth/Escape_on/'
plt.savefig(save_path , dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
