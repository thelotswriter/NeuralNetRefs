import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# Get the training and testing data
def fetch_data(selected_set):
    # CIFAR10 dataset
    if selected_set == '0':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
        return train_set, test_set, 10
    # KMNIST dataset
    elif selected_set == '1':
        transform = transforms.ToTensor()
        train_set = datasets.KMNIST(root='./data', train=True,
                                    download=True, transform=transform)
        test_set = datasets.KMNIST(root='./data', train=False,
                                   download=True, transform=transform)
        return train_set, test_set, 10
    # Default to CIFAR10 if an invalid dataset is chosen
    transform = transforms.ToTensor()
    train_set = datasets.KMNIST(root='./data', train=True,
                                download=True, transform=transform)
    test_set = datasets.KMNIST(root='./data', train=False,
                               download=True, transform=transform)
    return train_set, test_set, 10


# Define a basic convolutional neural network built for classifying images
class ConvClassifier(nn.Module):
    def __init__(self, dataset, output_size):
        super(ConvClassifier, self).__init__()
        # Get the first input (image) and target (label)
        x, y = dataset[0]
        # Determine the image channels (typically 1 or 3), height, and width
        c, h, w = x.size()
        print(f'c={c} h={h} w={w}')
        # Layers of our networks. Channels in, channels out, kernel size
        self.conv1 = nn.Conv2d(c, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        # Define pooling layers (like convolution layers, but take max value)
        self.pool = nn.MaxPool2d(2, stride=2)
        # Calculate height and width of the feature maps at the end of the convolution
        # and pooling layers
        adjusted_h = int(h / 4 - 3)
        adjusted_w = int(w / 4 - 3)
        # Add linear layers. Input for the first will be the product
        # of the adjusted height and weight and the final number of channels.
        # Basically, we want to know how many data points are in the stack of
        # feature maps
        self.linear_layers = nn.Sequential(
            nn.Linear(16 * adjusted_h * adjusted_w, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_size)
        )

    # Forward called when input is passed to the network
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = torch.flatten(out, 1)
        out = self.linear_layers(out)
        return out


# Train the network
def train(dataloader, model, loss_func, optim, device):
    size = len(dataloader.dataset)
    # Set the model into training mode
    model.train()
    train_losses = []
    # Training loop
    for batch, (X, y) in enumerate(dataloader):
        # Move input and target tensors to gpu if it was available
        X, y = X.to(device), y.to(device)
        # Predict and compute error
        yhat = model(X)
        # Calculate loss
        loss = loss_func(yhat, y)
        train_losses.append(loss.item())
        #  Do backprop and step down the gradient
        optim.zero_grad()
        loss.backward()
        optim.step()
    return train_losses


def test(dataloader, model, loss_func, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Put the net into evaluation mode
    model.eval()
    test_loss, correct = 0, 0
    # Loop through test data w/o calculating gradient (as we're not training)
    with torch.no_grad():
        for X, y in dataloader:
            # Ensure inputs and targets are on the appropriate device
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # Calculate the loss
            test_loss += loss_func(pred, y).item()
            # Increment correct if the prediction matched the target
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Calculate average loss
        test_loss /= num_batches
        # Average the number correct by the size of the dataset to get the accuracy
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def run(verbose=True, load=False, train_net=True):
    selection = input(f'What dataset would you like to use?\n'
                      f'0 - CIFAR10\n'
                      f'1 - KMNIST\n')
    # Get data, get model, get optimizer, train, test
    train_set, test_set, out_size = fetch_data(selection)
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvClassifier(train_set, out_size).to(device)
    if load:
        model.load_state_dict(torch.load(f'convclass{selection}.pth'))
    loss_func = nn.CrossEntropyLoss()
    if train_net:
        optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        # Determine how long to run the training based on user input
        epochs = input('How many epochs?\n')
        epochs = int(epochs)
        losses = []
        # Train and test the data based on the user's selection
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            losses.append(train(train_loader, model, loss_func, optim, device))
            test(test_loader, model, loss_func, device)
            print(losses[0])
        plt.plot(losses)
        plt.show()
        torch.save(model.state_dict(), f'convclass{selection}.pth')
        print(f'Saved Convolutional model state to convclass{selection}.pth')
    else:
        test(test_loader, model, loss_func, device)
