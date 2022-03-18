#!/usr/bin/env python
# coding: utf-8

    """ File to train neural network on baseball grip data. """

# In[3]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torch.optim as optim


# In[5]:

def train_model(save=False, epochs=1000):
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        loss_list.append(running_loss/len(train_loader))
        print(f'Finished epoch {epoch+1}.')

        print()

print('Finished Training')

def gen_loader(dir, batch_size=4, train=True):
    """Generates data loader for batch dataset

    Args:
        dir (string): directory containing data
        batch_size (int, optional): size of the batch; defaults to 4.
        train (bool, optional): training mode; defaults to True (in training mode).

    Returns:
        data_loader: augmented dataset
    """

    # define how we augment the data for composing the batch-dataset in train and test step
    transform = {
        'train': transforms.Compose([
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            #TODO if it is needed, add the random crop
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


# In[6]:


train_dir = os.path.join(os.getcwd(),'images/train')
test_dir = os.path.join(os.getcwd(),'images/test')

train_loader = gen_loader(data_dir=train_dir)
test_loader = gen_loader(data_dir=test_dir, train=False)

classes = (
    'changeup',
    'curveball',
    'fastball'
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


import matplotlib.pyplot as plt
import numpy as np

# In[8]:


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



# In[ ]:




# In[ ]:


PATH = 'trained_model.pth'
torch.save(net.state_dict(), PATH)

# In[ ]:


dataiter = iter(test_loader)
images, labels = dataiter.next()


# In[ ]:


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        # calculate outputs by running images through the network
        net.eval()
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the all test images: {100 * correct // total} %')


# In[ ]:


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1



# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


# In[ ]:


# import cv2
# from PIL import Image
# import numpy as np

# cam = cv2.VideoCapture(0)

# cv2.namedWindow("grip")

# img_counter = 0

# transform = transforms.Compose([
#             transforms.Resize([224,224]),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
# # prediction_list = []
# while True:
#     ret, frame = cam.read()
#     cv2.imshow("grip", frame)

#     if not ret:
#         print("failed to grab frame")
#         break
#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         ## SPACE pressed
# #         img_name = "changeup_wc_{}.png".format(img_counter)
# #         cv2.imwrite(img_name, frame)
# #         print("{} written!".format(img_name))
# #         img_counter += 1
        
#         im = Image.fromarray(frame*255)
#         transformed = transform(im)
#         normalized = transformed.float().unsqueeze(0)
#         output = net(normalized)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(output.data, 1)
#     #     print(classes[predicted])
#     #     print()
#     #     describe the type of font
#     #     to be used.
#     #     prediction_list.append(predicted)
#     #     avg_pred = int(round(np.mean(prediction_list[-10:]), 0))
#     #     font = cv2.FONT_HERSHEY_SIMPLEX

#     #     Use putText() method for
#     #     inserting text on video
#     #     cv2.putText(frame, 
#     #                 classes[avg_pred], 
#     #                 (50, 50), 
#     #                 font, 1, 
#     #                 (255, 0, 0), 
#     #                 2, 
#     #                 cv2.LINE_AA)
#         print(classes[predicted])  


# cam.release()

# cv2.destroyAllWindows()
