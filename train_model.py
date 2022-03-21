""" File to train neural network on baseball grip data. """

import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 64, 3)
        self.conv4 = nn.Conv2d(64, 192, 3)
        self.conv5 = nn.Conv2d(192, 384, 3)
        self.fc1 = nn.Linear(6144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        nn.Dropout()
        x = F.relu(self.fc1(x))
        nn.Dropout()
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Network:
    def __init__(self, model_path):
        self.path = model_path
        self.train_dir = os.path.join(os.getcwd(),'images/train')
        self.test_dir = os.path.join(os.getcwd(),'images/test')

        self.train_loader = self.gen_loader(dir=self.train_dir)
        self.test_loader = self.gen_loader(dir=self.test_dir, train=False)

        self.classes = (
            'changeup',
            'curveball',
            'fastball'
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(self, epochs=1000):
        net = Net().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        loss_list = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            loss_list.append(running_loss/len(self.train_loader))
            print(f'Finished epoch {epoch+1}.')

            print()

        print('Finished Training')

        PATH = self.path
        torch.save(net.state_dict(), PATH)

        return net


    def gen_loader(self, dir, batch_size=4, train=True):
        """Generates data loader for batch dataset

        Args:
            dir (string): directory containing data
            batch_size (int, optional): size of the batch; defaults to 4.
            train (bool, optional): training mode; defaults to True (in training mode).

        Returns:
            data_loader: augmented dataset

        Adapted from: kaggle.com/code/fatemehsharifi79/chess-pieces-detection-pytorch/notebook
        """

        transform = {
            'train': transforms.Compose([
                transforms.Resize([224,224]), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ]),
            'test': transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])
        }

        data = torchvision.datasets.ImageFolder(root=dir, transform=transform['train'] if train else transform['test'])
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=2)

        return data_loader

    def get_accuracy(self):
        dataiter = iter(self.test_loader)
        images, labels = dataiter.next()

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                # calculate outputs by running images through the network
                self.net.eval()
                outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the all test images: {100 * correct // total}%')
        print()


        correct_dict = {c: 0 for c in self.classes}
        total_dict = {c: 0 for c in self.classes}

        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                outputs = self.net(images)
                _, predictions = torch.max(outputs, 1)
                
                # count correct predictions
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_dict[self.classes[label]] += 1
                    total_dict[self.classes[label]] += 1

        # print accuracy for each class
        for c, correct in correct_dict.items():
            accuracy = 100 * float(correct) / total_dict[c]
            print(f'Predictions for {c} are {accuracy:.1f}% accurate.')