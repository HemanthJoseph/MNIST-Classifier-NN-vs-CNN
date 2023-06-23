# ======== A simple fully connected neural network using PyTorch ========= #

# ======= Imports ========= #
import torch  # entire PyTorch library
import torch.nn as nn  # Taking all neural network models
import torch.optim as optim  # optimization algos, SGD, Adam etc
import torch.nn.functional as F  # functions without params, ReLU, Tanh etc
from torch.utils.data import DataLoader  # Easier dataset mgmt, create mini batches etc
import torchvision.datasets as datasets  # Standard datasets from PyTorch
import torchvision.transforms as transforms  # transformation to perform on datasets
import torchsummary



# ======== Create a fully connected Network ==========#
class NN(nn.Module):  # inherits from nn.Module
    def __init__(self, input_size, num_classes):  # initialization. Input size is 784 as MNIST dataset has 28x28 size
        super(NN, self).__init__()  # calls the initialization method of parent class i.e. nn.Module
        self.fc1 = nn.Linear(input_size, 50)  # Hidden layer 1 has 50 nodes
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):  # Forward pass
        x = F.relu(self.fc1(x))  # First layer then relu on its output
        x = self.fc2(x)  # second layer
        return x


model = NN(784, 10)  # Input size is 784 as MNIST dataset has 28x28 size and 10 classes
x = torch.randn(64, 784)  # Initialize some x as input. 64 is number of examples running at time. Size of mini batch
print(model(x).shape)

# ============ Set device as cuda or CPU ============ #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Hyperparams =========== #
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# ======== Load Data =========== #
# using the inbuilt MNIST dataset in PyTorch

# initializing the training dataset
train_dataset = datasets.MNIST(root='../dataset/', train=True, transform=transforms.ToTensor(), download=True)
# Loading the training data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# initializing the testing dataset
test_dataset = datasets.MNIST(root='../dataset/', train=False, transform=transforms.ToTensor(), download=True)
# Loading the testing data
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# ========== Initialize the network ============ #
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# ========== Loss and Optimization ========== #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== Train the Network ============ #
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)  # 64 images by 1 input channel by height by width

        data = data.reshape(data.shape[0], -1)  # flattening the input

        # forward pass of NN
        scores = model(data)
        loss = criterion(scores, targets)

        # backward pass of NN
        optimizer.zero_grad()  # set all grads to zero for each batch
        loss.backward()

        # gradient descent or Adam step
        optimizer.step()


# ======= Check accuracy of our model ========== #
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data ")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():  # no need to compute any gradients in the calculations
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

model.to(device)
torchsummary.summary(model, (1, 784))
