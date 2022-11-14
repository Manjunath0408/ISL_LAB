import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchmetrics import Precision, Accuracy, F1Score, Recall

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Define model


class NeuralNetwork(nn.Module):
  def __init__(self, config, s):
    super().__init__()
    classes = 10

    self.layers = nn.ModuleList()
    for c in config:
      conv = nn.Conv2d(in_channels = c[0], out_channels = c[1], kernel_size = c[2], stride = c[3])
      self.layers.append(conv)

    self.layers.append(nn.ReLU())
    for layer in self.layers:
      s = layer(s)
    s = s.shape
    size = s[1]*s[2]*s[3]
    self.layers.append(nn.Flatten())
    self.layers.append(nn.Linear(size, classes))
    self.layers.append(nn.Softmax(dim = 1))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

#############################
def loss_fn(y_pred, y_ground):
  v = -torch.mul(y_ground, torch.log(y_pred + 1e-4))
  v = v.sum()
  return v

def get_lossfn_and_optimizer(mymodel):
  optimizer = torch.optim.SGD(mymodel.parameters(), lr=1e-3)
  return loss_fn, optimizer

def load_data():
  training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
  )
  test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
  )
  return training_data, test_data

#############################


def create_dataloaders(training_data, test_data, batch_size=64):
  train_dataloader = DataLoader(training_data, batch_size=batch_size)
  test_dataloader = DataLoader(test_data, batch_size=batch_size)
  return train_dataloader, test_dataloader

#############################


def get_model(config, sample):
  model = NeuralNetwork(config, sample)
  return model

# def _train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train_network(train_dataloader, model, loss_fn, optimizer, epoch = 1):
  size = len(train_dataloader.dataset)
  model.train()
  for e in range(epoch):
    cnt = 0
    for X,y in train_dataloader:
      cnt += 1
      y_pred = model(X)
      y_ground = torch.nn.functional.one_hot(torch.tensor(y, dtype = int), num_classes = 10)

      loss = loss_fn(y_pred, y_ground)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if cnt%100 == 0:
        loss, current = loss.item(), cnt * len(X)
        print(f"loss: [{loss:>7f}] [{current:>5d}]/[{size:>5d}]")


# def _test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(
#         f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_network(test_dataloader, model, loss_fn):
  accuracy = Accuracy()
  f1score = F1Score()
  precision = Precision()
  recall = Recall()
  model.eval()
  with torch.no_grad():
    for X,y in train_dataloader:
      y_pred = model(X)
      y_ground = torch.nn.functional.one_hot(torch.tensor(y, dtype = int), num_classes = 10)
      loss = loss_fn(y_pred, y_ground)
      acc = accuracy(y_pred, y_ground)
      f1 = f1score(y_pred, y_ground)
      prec = precision(y_pred, y_ground)
      rec = recall(y_pred, y_ground)
    
    accuracy = accuracy.compute().item()*100
    precision = precision.compute().item()
    f1score = f1score.compute().item()
    recall = recall.compute().item()
    print("\n*************************")
    print("Accuracy over all data: ", accuracy)
    print("Precision over all data: ", precision)
    print("Recall over all data: ", recall)
    print("F1Score over all data: ", f1score)
    print("*************************")
    return accuracy, precision,recall, f1score


def train(train_dataloader, test_dataloader, model1, loss_fn1, optimizer1, epochs=5):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_network(train_dataloader, model1, loss_fn1, optimizer1)
        test_network(test_dataloader, model1, loss_fn1)
    print("Done!")
    return model1


def save_model(model1, mypath="model.pth"):
    torch.save(model1.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


def load_model(mypath="model.pth"):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    return model


def sample_test(model1, test_data):
    model1.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model1(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
