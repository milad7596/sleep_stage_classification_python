import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import json

# Set the random seed for reproducibility
torch.manual_seed(123)

# Define the transform for preprocessing the data
transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hyperparameters
num_classes = 6
learning_rate = 0.05
batch_size = 128
num_epochs = 50

name_solver = "SGD"

data_path = r"D:/Paper (Sleep stage)/sleep-edf-database-expanded-1.0.0/" \
            r"sleep-telemetry/All IMAGES"  # Replace with the path to your dataset directory

# Read images from the directory
dataset = datasets.ImageFolder(data_path, transform=transform)

# Split the dataset into training and testing sets
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=1)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Define the dataloaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the ResNet-18 model
model = models.resnet18(weights=None, num_classes=num_classes)

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9 ) #momentum=0.9 SGD

# Lists to store the loss values and accuracies for plotting
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
statistics = ""
# Training loop
for epoch in range(num_epochs):
    train_loss = 0.0
    correct = 0
    total = 0

    model.train()  # Set the model to training mode

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    train_loss /= len(train_loader)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)

    # Evaluate the model on the test set
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Compute test statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = 100 * correct / total
    test_loss /= len(test_loader)
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)
    statistics += f"Epoch [{epoch + 1}/{num_epochs}] - Train Accuracy: {train_accuracy:.2f}% - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%\n"
    # Print the training and testing statistics for each epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Accuracy: {train_accuracy:.2f}% - Train Loss: {train_loss:.4f}"
          f" - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_predicted)
label_names = dataset.classes
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(9, 7))
sns.heatmap(cm_percent, annot=True, fmt=".2%", cbar=False, cmap="Blues")  # cmap="Blues"
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{name_solver} solver- learning:{learning_rate} - batch:{batch_size} - "
          f" num epoch:{num_epochs} - num class{num_classes} ")
plt.xticks(np.arange(num_classes) + 0.5, label_names, rotation='vertical')
plt.yticks(np.arange(num_classes) + 0.5, label_names, rotation='horizontal')
plt.savefig(f"{learning_rate} -{batch_size} - {num_epochs} confusion_matrix.jpg", dpi=300)

plt.figure(figsize=(9, 8))
sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues")  #
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f" {name_solver} solver- learning:{learning_rate} - batch:{batch_size} - "
          f" num epoch:{num_epochs} - num class{num_classes} ")
plt.xticks(np.arange(num_classes) + 0.5, label_names, rotation='vertical', fontsize=15)
plt.yticks(np.arange(num_classes) + 0.5, label_names, rotation='horizontal', fontsize=15)
plt.savefig(f"{learning_rate} - {batch_size} - {num_epochs} confusion_matrix1.jpg", dpi=300)

# Save test and training data as a note
# note = f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%\n" \
       f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%"

classification_rep = classification_report(all_labels, all_predicted,
                                           target_names=label_names, zero_division=1)

parameters = f"number of  classes: {num_classes}\n\n" \
             f"Learning rate: {learning_rate}\n\n" \
             f"batch size: {batch_size}\n\n" \
             f"number of epochs: {num_epochs}\n\n"

sum_accuracy = sum([cm_percent[i][i] for i in range(len(cm_percent))])
cm_acc = f"Accuracy of R : {round(cm_percent[0][0] * 100, 2)}\n\n" \
         f"Accuracy of S1 : {round(cm_percent[1][1] * 100, 2)}\n\n" \
         f"Accuracy of S2 : {round(cm_percent[2][2] * 100, 2)}\n\n" \
         f"Accuracy of S3 : {round(cm_percent[3][3] * 100, 2)}\n\n" \
         f"Accuracy of S4 : {round(cm_percent[4][4] * 100, 2)}\n\n" \
         f"Accuracy of W : {round(cm_percent[5][5] * 100, 2)}\n\n" \
         f"Overal accuracy : {round((sum_accuracy * 100)/6 ,2)}\n\n"

print(classification_rep)

with open(f"{learning_rate} - {batch_size} - {num_epochs} - {num_classes} "
          f"classification_report.txt", "w") as file:
    file.write(name_solver +"\n"+ parameters + "\n" + classification_rep + "\n""\n"
               + cm_acc + "\n""\n" + statistics)

# Plotting the loss and accuracy curves
plt.figure(figsize=(12, 6))
plt.plot(test_accuracies, label='Validation Accuracy')
plt.plot(test_losses, label='Validation Loss')
plt.title(f"{name_solver} solver - Learning: {learning_rate} - batch: {batch_size} - "
          f" num epoch: {num_epochs} - num class: {num_classes}")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(visible=True)
plt.savefig(f"{learning_rate} - {batch_size} - {num_epochs} - {num_classes} - "
            f"Validation plots.jpg", dpi=300)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()

plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(train_losses, label='Training Loss')
plt.title(f"{name_solver} solver - Learning: {learning_rate} - batch: {batch_size} - "
          f"num epoch: {num_epochs}- num class: {num_classes}")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(visible=True)
plt.savefig(f"{learning_rate} - {batch_size} - {num_epochs} - {num_classes} - "
            f" Test plots.jpg", dpi=300)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()

# Save parameters to a file using JSON
with open(f'{learning_rate} - {batch_size} - {num_epochs} - {num_classes}'
          f' test_accuracies.json', 'w') as file:
    json.dump(test_accuracies, file)

with open(f'{learning_rate} - {batch_size} - {num_epochs} - {num_classes} '
          f'test_losses.json', 'w') as file:
    json.dump(test_losses, file)

with open(f'{learning_rate} - {batch_size} - {num_epochs} - {num_classes}'
          f' train_accuracies.json', 'w') as file:
    json.dump(train_accuracies, file)

with open(f'{learning_rate} - {batch_size} - {num_epochs} - {num_classes}'
          f' train_losses.json', 'w') as file:
    json.dump(train_losses, file)



# plt.show()

# Load parameters from the saved file
# with open('parameters.json', 'r') as file:
#     loaded_parameters = json.load(file)