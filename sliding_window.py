import cv2
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import classify_grip

def sliding_window(image, window=(300, 300), step=32):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield (x, y, image[y:y+window[1], x:x+window[0]])


image = cv2.imread('images/grips/augmented/test/curveball/curveball_487894806.jpg-0.jpg')

resized = cv2.resize(image, (int(image.shape[1]*2), int(image.shape[0]*2)))

net = classify_grip.Net()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = 'beast_model.pth'
net.load_state_dict(torch.load(PATH, map_location=torch.device(device)))

transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

highest_p = []
for (x, y, window) in sliding_window(resized):
    if (x + 300 > resized.shape[1]) or (y + 300 > resized.shape[0]):
        continue
    cv2.rectangle(resized, (x, y), (x+300, y+300), (255, 0, 0), 2)
    # cv2.rectangle(resized, (x, y), (x + window[0], y + window[1]), (255, 0, 0), 3)
    cv2.imshow("test", resized)
    cv2.waitKey(1)
    time.sleep(0.05)
    im = Image.fromarray(window*255)
    transformed = transform(im)
    normalized = transformed.float().unsqueeze(0)
    output = net(normalized)

    prob = torch.softmax(output.data, 1)
    top_p, top_class = prob.topk(1, 1)

    




