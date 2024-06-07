
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models import resnet18
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 3
batch_size = 64
learning_rate = 0.001
validation_split = 0.2
chosen_classes = [2, 3]
dataset_path = '/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Mnist'  # Define the path where MNIST will be saved


# Transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # MNIST is grayscale, ResNet expects 3 channels
    transforms.Resize((224, 224)),  # ResNet-18 expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizing MNIST dataset
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform, download=True)

# Filter the dataset for classes 0 and 1
train_indices = [i for i, (img, label) in enumerate(train_dataset) if label in chosen_classes]
train_dataset = Subset(train_dataset, train_indices)

# Split the dataset into training and validation sets
train_size = int((1 - validation_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)

# Load the test set
test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transform, download=True)
test_indices = [i for i, (img, label) in enumerate(test_dataset) if label in chosen_classes]
test_dataset = Subset(test_dataset, test_indices)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("Train {}, Val {}, Test {}".format(len(train_subset), len(val_subset), len(test_dataset)))

# Initialize ResNet-18 model
model = resnet18(weights='DEFAULT')
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
classifier = nn.Linear(model.fc.in_features, 2)

# Define prototypes as learnable parameters
prototype_0 = nn.Parameter(torch.randn(1, 3, 224, 224, requires_grad=True, device=device))
prototype_1 = nn.Parameter(torch.randn(1, 3, 224, 224, requires_grad=True, device=device))
prototypes = [prototype_0, prototype_1]

# Add prototypes to a list of parameters
model_params = list(feature_extractor.parameters()) + list(classifier.parameters())

feature_extractor = feature_extractor.to(device)
classifier = classifier.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_params, lr=learning_rate)
# optimizer = optim.Adam(model_params, lr=learning_rate)

# Training and validation loop
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        feature_extractor.train()
        classifier.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            labels[labels == chosen_classes[0]] = 0
            labels[labels == chosen_classes[1]] = 1
            # Forward pass
            output_features = feature_extractor(images)
            outputs = classifier(output_features.squeeze())
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}') #, Prot_Loss: {loss_prot.item():.4f}

        # Validate the model
        feature_extractor.eval()
        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                output_features = feature_extractor(images)
                outputs = classifier(output_features.squeeze())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Validation accuracy after epoch {epoch + 1}: {accuracy:.2f} %')

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Test the model
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output_features = feature_extractor(images)
            outputs = classifier(output_features.squeeze())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total} %')

# Train and validate the model
best_model = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Test the best model
test(best_model, test_loader)

# Save the best model checkpoint
torch.save(best_model.state_dict(), 'best_resnet18_mnist_2_classes.pth')


def train_prototypes_ce(model, prototypes, criterion, num_epochs):
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Ensure optimizer only updates the prototypes
    optimizer = optim.Adam(prototypes, lr=learning_rate)

    for epoch in range(num_epochs):

        # Add prototypes to the batch
        prototype_labels = torch.tensor([0, 1], device=device)
        prototype_images = torch.cat(prototypes, dim=0)

        output_features = feature_extractor(prototype_images)
        outputs = classifier(output_features.squeeze())
        loss = criterion(outputs, prototype_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item() / len(prototype_labels)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Prototype Loss: {epoch_loss:.4f}, Preds: {outputs}')

    return prototypes

def train_prototypes_mse(model, prototypes, train_loader, val_loader, num_epochs):
    best_mse = 1000.0
    best_prototypes = copy.deepcopy(prototypes)
    optimizer = optim.Adam(prototypes, lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            labels[labels == chosen_classes[0]] = 0
            labels[labels == chosen_classes[1]] = 1

            prototype_loss = 0
            for img, label in zip(images, labels):
                if label == 0:
                    prototype_loss += F.mse_loss(img, prototype_0)
                elif label == 1:
                    prototype_loss += F.mse_loss(img, prototype_1)
            prototype_loss /= len(labels)

            # Backward and optimize
            optimizer.zero_grad()
            prototype_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Prot Loss: {prototype_loss.item():.4f}') #, Prot_Loss: {loss_prot.item():.4f}

        with torch.no_grad():
            # correct = 0
            total = 0
            for images, labels in val_loader:
                prototype_loss = 0
                for img, label in zip(images, labels):
                    if label == 0:
                        prototype_loss += F.mse_loss(img, prototype_0)
                    elif label == 1:
                        prototype_loss += F.mse_loss(img, prototype_1)
                prototype_loss /= len(labels)
                total += prototype_loss
            total /= len(val_loader)
            print(f'Validation mse after epoch {epoch + 1}: {total:.2f} %')

            # Save the best model
            if total < best_mse:
                best_mse = total
                best_prototypes = copy.deepcopy(prototypes)
    return best_prototypes

prototypes_ce = train_prototypes_ce(model, [prototype_0, prototype_1], criterion, num_epochs=200)
prototypes_mse = train_prototypes_mse(model, [prototype_0, prototype_1], train_loader, val_loader, num_epochs=10)


def find_closest_and_furthest_points(prototypes, dataloader, num_points=5):
    closest_points = {0: [], 1: []}
    furthest_points = {0: [], 1: []}
    closest_distances = {0: [], 1: []}
    furthest_distances = {0: [], 1: []}
    prototypes = [prototypes['prototype_0'], prototypes['prototype_1']]

    with torch.no_grad():
        for images, labels in dataloader:
            labels[labels == chosen_classes[0]] = 0
            labels[labels == chosen_classes[1]] = 1
            for img, label in zip(images, labels):
                # label = label.item()
                for pi in range(len(prototypes)):

                    dist = torch.norm(img - prototypes[pi])

                    # For closest points
                    if len(closest_distances[pi]) < num_points:
                        closest_distances[pi].append(dist)
                        closest_points[pi].append(img)
                    else:
                        max_closest_dist_idx = np.argmax(closest_distances[pi])
                        if dist < closest_distances[pi][max_closest_dist_idx]:
                            closest_distances[pi][max_closest_dist_idx] = dist
                            closest_points[pi][max_closest_dist_idx] = img

                    # For furthest points
                    if len(furthest_distances[pi]) < num_points:
                        furthest_distances[pi].append(dist)
                        furthest_points[pi].append(img)
                    else:
                        min_furthest_dist_idx = np.argmin(furthest_distances[pi])
                        if dist > furthest_distances[pi][min_furthest_dist_idx]:
                            furthest_distances[pi][min_furthest_dist_idx] = dist
                            furthest_points[pi][min_furthest_dist_idx] = img

    return closest_points, furthest_points
def plot_prototypes_and_points(prototypes, closest_points, furthest_points, title):
    prototypes = [prototypes['prototype_0'], prototypes['prototype_1']]
    fig, axes = plt.subplots(4, 6, figsize=(18, 9))  # 3 rows: prototypes, closest, furthest

    for i in range(2):
        # Plot prototypes
        prototype = prototypes[i].detach().cpu().squeeze().permute(1, 2, 0)
        axes[i*2, 0].imshow(prototype, cmap='gray')
        axes[i*2, 0].set_title(f'Prototype {i}')
        axes[i*2, 0].axis('off')
        axes[i*2+1, 0].axis('off')

        # Plot closest points
        for j, img in enumerate(closest_points[i]):
            img = img.cpu().squeeze().permute(1, 2, 0)
            axes[i*2, j + 1].imshow(img, cmap='gray')
            axes[i*2, j + 1].set_title(f'Closest {j + 1}')
            axes[i*2, j + 1].axis('off')

        # Plot furthest points
        for j, img in enumerate(furthest_points[i]):
            img = img.cpu().squeeze().permute(1, 2, 0)
            axes[1+i*2, j + 1].imshow(img, cmap='gray')
            axes[1+i*2, j + 1].set_title(f'Furthest {j + 1}')
            axes[1+i*2, j + 1].axis('off')
    plt.title(title)
    # plt.tight_layout()
    plt.show()

prototypes = { 'prototype_0': prototypes_ce[0].detach().cpu(), 'prototype_1': prototypes_ce[1].detach().cpu()}
closest_points, furthest_points = find_closest_and_furthest_points(prototypes, train_loader)
plot_prototypes_and_points(prototypes, closest_points, furthest_points, "CE Prototypes")

prototypes = { 'prototype_0': prototypes_mse[0].detach().cpu(), 'prototype_1': prototypes_mse[1].detach().cpu()}
closest_points, furthest_points = find_closest_and_furthest_points(prototypes, train_loader)
plot_prototypes_and_points(prototypes, closest_points, furthest_points, "MSE Prototypes")
