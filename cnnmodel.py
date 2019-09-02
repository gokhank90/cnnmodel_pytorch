import torch
import torchvision

n_epochs = 2
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
plot_interval = 0
createLasttrainLoss = []
createLosstest = []

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/gkhnkrhmt/datasets', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)  # batch parametresi

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/gkhnkrhmt/datasets', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)  # parametreleri burda verdik batch için

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)  # 1000 examples of 28x28 pixels in grayscale (i.e. no rgb channels, hence the one).

# Görsellik için:
import matplotlib.pyplot as plt

# fig = plt.figure()
"""for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# bunu ben ekledim.
def plot_it(n_epochs, createLasttrainLoss, createLosstest):
    plt.plot(range(n_epochs), createLasttrainLoss, 'r--')
    plt.plot(range(n_epochs), createLosstest, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


# we can think of the torch.nn layers
# which contain trainable parameters while torch.nn.functional are purely functional.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        a = self.conv1.weight
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch, plot_interval, createLasttrainLoss):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)  # network'e datayı yollayarak datayı eğitiyor.
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')
            train_losses.append(loss.item())
            plot_interval += 1
            print(plot_interval)

    createLasttrainLoss.append(train_losses[len(train_losses) - 1])
    return createLasttrainLoss


def test(createLosstest):
    network.eval()
    test_loss = 0
    correct = 0
    test_lossCounter = 1
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print(test_lossCounter)
    print("Testloses boyuutu=" + str(len(test_losses)))
    createLosstest.append(test_losses[len(test_losses) - 1])
    return createLosstest


for epoch in range(1, n_epochs + 1):
    createLasttrainLoss = train(epoch, plot_interval, createLasttrainLoss)
    createLosstest = test(createLosstest)

plot_it(n_epochs, createLasttrainLoss, createLosstest)
