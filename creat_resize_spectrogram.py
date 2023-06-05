import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage.transform import resize
import pickle


channel= "ST-Fpz_Cz"
# Specify the file path where the dictionary is saved
file_path = 'D:/Paper (Sleep stage)/sleep-edf-database-expanded-1.0.0/' \
            f'sleep-telemetry/data_pickle/{channel}.pickle'

# Open the file in binary mode
with open(file_path, 'rb') as file:
    # Use pickle to load the dictionary object
    my_dict = pickle.load(file)

# sbj_EEGs_Fpz_Cz = {}  # Assuming you have defined sbj_EEGs_Fpz_Cz.W
stages = "S2"

# def save_and_resize_images(start, end, image_folder):
def save_and_resize_images(start, end,step, image_folder):
    # Create the directory if it doesn't exist
    os.makedirs(image_folder, exist_ok=True)

    # for i in range(start, end + 1):
    for i in range(start, end + 1, step):
        m = my_dict[stages][i - 1]

        file_name = f'im_resize_128-128_{stages}_{i}.jpg'
        file_path = os.path.join(image_folder, file_name)
        img_path = os.path.join(image_folder, file_name)

        # Plot spectrogram
        f, t, Sxx = signal.spectrogram(m, fs=100, mode='magnitude', window='hann',
                                       scaling='density', nperseg=256, noverlap=128)

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', cmap='hot')
        plt.colorbar().remove()
        plt.axis('off')
        plt.title('')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(file_path, dpi=200)
        plt.close()

        # Resize image
        im_read = plt.imread(file_path)
        im_resize = resize(im_read, (128, 128), anti_aliasing=True)
        plt.imsave(img_path, im_resize, cmap='gray', dpi=200)

        # Optional display
        # plt.imshow(im_resize)
        # plt.show()


image_folder = r'D:/Paper (Sleep stage)/sleep-edf-database-expanded-1.0.0' \
               f'/sleep-telemetry/test thesis/tesstttt/images/{stages}'  # Specify the directory where you want to save the images

save_and_resize_images(1, 19851, 6,image_folder)  # Call the function and pass the start, end, and image_folder arguments

## Save and resize images for the respective ranges
# save_and_resize_images(1, len(read_data_Fpz_Cz.sbj_EEGs_Fpz_Cz[stages]))

## Save images every ex 5
# save_and_resize_images(1, len(read_data_Fpz_Cz.sbj_EEGs_Fpz_Cz[stages]), 2, image_folder)

##  Save images between 5 to 10 range ex
# save_and_resize_images(5,10)
