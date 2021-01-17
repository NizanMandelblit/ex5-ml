from gcommand_dataset import GCommandLoader
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lay1 = nn.Sequential(nn.Conv2d(1, 18, kernel_size=5, stride=1, padding=2),
                                  nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.lay2 = nn.Sequential(nn.Conv2d(18, 58, kernel_size=5, stride=1, padding=2),
                                  nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.lay3 = nn.Sequential(nn.Conv2d(58, 12, kernel_size=5, stride=1, padding=2),
                                  nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.d = nn.Dropout()
        self.fc0 = nn.Linear(4600, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 30)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        #F.log_softmax(x, dim=1)


def train(model, optimizer, train_loader):
    model.train()
    for epoch in range(10):
        test_loss = 0
        #correct = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            """
            # for testing
            test_loss += F.nll_loss(output, labels, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).cpu().sum()
            # end testing
            """
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
        test_loss /= len(train_loader.dataset)
        # print(epoch)
        # print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(train_loader.dataset),100. * correct / len(train_loader.dataset)))


def valid(model, valid_loader):
    model.eval()
    for epoch in range(10):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).cpu().sum()
        test_loss /= len(valid_loader.dataset)
        print(epoch)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                   len(test_loader.dataset),
                                                                                   100. * correct / len(
                                                                                   test_loader.dataset)))


def test(model, test_loader):
    model.eval()
    for epoch in range(10):
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                print(pred)


def main():
    train_set=GCommandLoader('gcommands/train')
    test_set = GCommandLoader('gcommands/test')
    valid_set = GCommandLoader('gcommands/valid')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=110, shuffle=True,
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        test_set, batch_size=110, shuffle=True,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=110, shuffle=False,
        pin_memory=True)
    mymodel=MyModel()
    optimizer = optim.SGD(mymodel.parameters(), lr=0.001, momentum=0.9)
    train(mymodel, optimizer, train_loader)
    valid(mymodel, valid_loader)
    test(mymodel, test_loader)




if __name__ == "__main__":
    main()