import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from customDatasets import CSVDataset


# Get the training and testing data
def fetch_data(selected_set):
    # FashionMNIST
    if selected_set == '0':
        train_set = datasets.FashionMNIST(
            root='data', train=True, download=True, transform=ToTensor(),
        )
        test_set = datasets.FashionMNIST(
            root='data', train=False, download=True, transform=ToTensor(),
        )
        print('FashionMNIST type:')
        print(type(train_set))
        return train_set, test_set, 28 * 28, 10
    # Iris dataset
    elif selected_set == '1':
        iris_dataset = CSVDataset('https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv')
        train_set, test_set = iris_dataset.get_split_sets()
        print('Iris type:')
        print(type(train_set))
        return train_set, test_set, 4, 3
    # Default dataset. FashionMNIST, in this case
    else:
        train_set = datasets.FashionMNIST(
            root='data', train=True, download=True, transform=ToTensor(),
        )
        test_set = datasets.FashionMNIST(
            root='data', train=False, download=True, transform=ToTensor(),
        )
        return train_set, test_set, 28 * 28, 10


# Defined a basic Multilayer Perceptron for classification.
class MLP(nn.Module):
    # Initializes the net
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, n_outputs)
        )

    # Called when input (x) is run through the net.
    # Returns predicted results
    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits


# Trains the net. Takes the model (net), criterion/loss function,
# optimizer, and the device being used
def train(dataloader, model, loss_func, optim, device):
    size = len(dataloader.dataset)
    # Set the model into training mode
    model.train()
    # Training loop
    for batch, (X, y) in enumerate(dataloader):
        # Move input and target tensors to gpu if it was available
        X, y = X.to(device), y.to(device)
        # Predict and compute error
        yhat = model(X)
        # Calculate loss
        loss = loss_func(yhat, y)
        #  Do backprop and step down the gradient
        optim.zero_grad()
        loss.backward()
        optim.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Tests the network to check performance
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


# Runs the multilayer perceptron classification code
# verbose - print additional information
# load - Load a former model or start a new one
# train_net - Train the network or just run on test data
def run(verbose=True, load=False, train_net=True):
    if verbose:
        print(f'Verbose: {verbose}')
        print(f'Load: {load}')
        print(f'Train: {train_net}')
    # Set up which dataset will be used
    set_choice = input('Select a dataset:\n'
                       '0 - FashionMNIST\n'
                       '1 - Iris\n')
    # Get the selected dataset as test and train sets with the necessary
    # Input and output size for the network
    train_set, test_set, input_size, output_size = fetch_data(set_choice)
    batch_size = 64
    # Set up DataLoaders for iterating through the data
    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)
    if verbose:
        for X,y in test_dataloader:
            print(f'Input Shape (X): {X.shape}')
            print(f'Target Shape (y): {y.shape}')
            break
    # Use gpu, if available. Otherwise we'll settle for cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f'Using {device}.')
    # Create the net
    model = MLP(input_size, output_size).to(device)
    if verbose:
        print("Model:")
        print(model)
    # Load save data, if selected
    if load:
        model.load_state_dict(torch.load(f'mlp{set_choice}.pth'))
    # Initialize the loss function. For multiple, categorical outputs we use CrossEntropyLoss
    loss_func = nn.CrossEntropyLoss()
    if train_net:
        # Set up the optimizer. Adam is a fairly standard option
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Determine how long to run the training based on user input
        epochs = input('How many epochs?\n')
        epochs = int(epochs)
        # Train and test the data based on the user's selection
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}')
            train(train_dataloader, model, loss_func, optim, device)
            test(test_dataloader, model, loss_func, device)
        torch.save(model.state_dict(), f'mlp{set_choice}.pth')
        print(f'Saved MLP model state to mlp{set_choice}.pth')
    else:
        # Run the test function without training
        test(test_dataloader, model, loss_func, device)
