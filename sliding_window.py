import cv2
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

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

highest_p = 0
top_window = []
df = pd.DataFrame(columns=[*range((resized.shape[1]-300)//32)])
row = []
append_flag = False
for i, (x, y, window) in enumerate(sliding_window(resized)):

    if (x + 300 > resized.shape[1]) or (y + 300 > resized.shape[0]):
        if append_flag:
            df.loc[i] = row[:-1]
        append_flag = False
        row = []
        count = 0
        continue

    append_flag = True
    # cv2.rectangle(resized, (x, y), (x+300, y+300), (255, 0, 0), 2)
    # # cv2.rectangle(resized, (x, y), (x + window[0], y + window[1]), (255, 0, 0), 3)
    # cv2.imshow("test", resized)
    # cv2.waitKey(1)
    time.sleep(0.05)
    im = Image.fromarray(window*255)
    transformed = transform(im)
    normalized = transformed.float().unsqueeze(0)
    output = net(normalized)

    prob = torch.softmax(output.data, 1)
    top_p, top_class = prob.topk(1, 1)

    if top_p > highest_p:
        highest_p = top_p
        top_window = ((x, y), (x+300, y+300))
    row.append(top_p.numpy()[0][0])
    

# df.to_csv('output.csv')

df = df.applymap(lambda x: 0.0 if (x < 0.9) else x)

high_score = 0
high_xy = None

for index in range(df.shape[0]-2):
    for i in range(df.shape[1]-2):
        score = df.iloc[index:3+index, i:3+i].values.mean()
        print(score)
        if score > high_score:
            high_score = score
            high_xy = ((i-1)*32, (index-1)*32)
        # cv2.rectangle(resized, (i*32, index*32), (364+(i*32), 364+(index*32)), (0, 255, 0), 2)
        # cv2.imshow("test", resized)
        # cv2.waitKey(1)
        # time.sleep(0.05)

# print(high_xy)
# cv2.rectangle(resized, high_xy, (high_xy[0]+364, high_xy[1]+364), (0, 255, 0), 2)
# cv2.imshow("test", resized)
# cv2.waitKey(10000)

classify_window = resized[high_xy[1]:high_xy[1]+428, high_xy[0]:high_xy[0]+428]
cv2.imwrite('test_image.png', classify_window)
        




