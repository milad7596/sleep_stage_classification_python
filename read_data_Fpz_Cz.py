import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

sbj_EEGs_Fpz_Cz = {
    'W': [],
    'R': [],
    'S1': [],
    'S2': [],
    'S3': [],
    "S4": []
}

channel= "ST-Fpz_Cz"
# read data
edf_files = [file for file in os.listdir() if file.endswith(".edf")]

for sbj in range(0, len(edf_files), 2):
    sample_001 = edf_files[sbj]  # "SC4001E0-PSG.edf"
    sample_001_hy = edf_files[sbj + 1]  # "SC4001EC-Hypnogram.edf"
    raw = mne.io.read_raw_edf(sample_001, preload=True)
    info = raw.info

    data_hy = mne.read_annotations(sample_001_hy)
    # header_time = data_hy.read_annotations()

    annotation = data_hy.description
    classes = np.unique(annotation)

    # Access the channel names
    channel_names = raw.ch_names
    # print(f"Channels: {channel_names}")

    # Print the first channel
    channel_index = 0
    channel_data = raw.get_data(picks=[channel_index])
    # print(f"Channel_Selected: {channel_names[channel_index]}")
    # print(f"Channel data: {channel_data}")

    len_annotations = len(data_hy.onset)
    num_annotations = len_annotations - 1

    for i in range(num_annotations):
        k = int(data_hy.onset[i] * raw.info['sfreq'])
        j = int(data_hy.onset[i + 1] * raw.info['sfreq'])
        a = raw.get_data(picks=[channel_index], start=k,
                         stop=j)[0]

        if annotation[i] == "Sleep stage W":
            num_arrays = len(a) // 3000  # Number of arrays with 3000 values each
            for h in range(num_arrays):
                start_index = h * 3000  # Starting index of the current array
                end_index = (h + 1) * 3000  # Ending index of the current array
                sub_array = a[start_index:end_index]  # Extract the sub-array with 3000 values
                sbj_EEGs_Fpz_Cz['W'].append(sub_array)

        elif annotation[i] == "Sleep stage R":
            num_arrays = len(a) // 3000  # Number of arrays with 3000 values each
            for h in range(num_arrays):
                start_index = h * 3000  # Starting index of the current array
                end_index = (h + 1) * 3000  # Ending index of the current array
                sub_array = a[start_index:end_index]  # Extract the sub-array with 3000 values
                sbj_EEGs_Fpz_Cz['R'].append(sub_array)

        elif annotation[i] == "Sleep stage 1":
            num_arrays = len(a) // 3000  # Number of arrays with 3000 values each
            for h in range(num_arrays):
                start_index = h * 3000  # Starting index of the current array
                end_index = (h + 1) * 3000  # Ending index of the current array
                sub_array = a[start_index:end_index]  # Extract the sub-array with 3000 values
                sbj_EEGs_Fpz_Cz['S1'].append(sub_array)

        elif annotation[i] == "Sleep stage 2":
            num_arrays = len(a) // 3000  # Number of arrays with 3000 values each
            for h in range(num_arrays):
                start_index = h * 3000  # Starting index of the current array
                end_index = (h + 1) * 3000  # Ending index of the current array
                sub_array = a[start_index:end_index]  # Extract the sub-array with 3000 values
                sbj_EEGs_Fpz_Cz['S2'].append(sub_array)

        elif annotation[i] == "Sleep stage 3":
            num_arrays = len(a) // 3000  # Number of arrays with 3000 values each
            for h in range(num_arrays):
                start_index = h * 3000  # Starting index of the current array
                end_index = (h + 1) * 3000  # Ending index of the current array
                sub_array = a[start_index:end_index]  # Extract the sub-array with 3000 values
                sbj_EEGs_Fpz_Cz['S3'].append(sub_array)

        elif annotation[i] == "Sleep stage 4":
            num_arrays = len(a) // 3000  # Number of arrays with 3000 values each
            for h in range(num_arrays):
                start_index = h * 3000  # Starting index of the current array
                end_index = (h + 1) * 3000  # Ending index of the current array
                sub_array = a[start_index:end_index]  # Extract the sub-array with 3000 values
                sbj_EEGs_Fpz_Cz['S4'].append(sub_array)


# Assuming your dictionary is named 'my_dict'
# Specify the file path where you want to save the dictionary
file_path = 'D:/Paper (Sleep stage)/sleep-edf-database-expanded-1.0.0/' \
            f'sleep-telemetry/data_pickle/{channel}.pickle'

# Open the file in binary mode
with open(file_path, 'wb') as file:
    # Use pickle to serialize and save the dictionary object
    pickle.dump(sbj_EEGs_Fpz_Cz, file)



print(f"Channels: **{channel_names}**")
print(f"Channel_Selected:+++{channel_names[channel_index]}+++")
# times = raw.times
# num_samples= 3000
#
# # Plot the desired segment of EEG data
# plt.plot(times[:num_samples], channel_data[0][:num_samples])
# plt.xlabel('Time (s)')
# plt.ylabel('EEG')
# plt.title('EEG Data (Sleep stage 4, 30 seconds)')
# plt.show()
