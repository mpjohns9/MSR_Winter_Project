
""" Sliding window method for object detection of baseball. """

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from train_model import Net

class slidingWindow:
    def __init__(self, model_dir):
        """The init function."""

        self.net = Net()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_dir:
            self.net.load_state_dict(torch.load(model_dir, map_location=torch.device(self.device)))

        self.transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

    def sliding_window(image, window=(300, 300), step=32):
        """Creates sliding window over image with step size.

        Args:
            image: image to slide over
            window: window size; defaults to (300, 300)
            step: step size; defaults to 32

        Yields:
            x, y, image: x, y coords of upper left window and 
                         portion of image in window
        """
        for y in range(0, image.shape[0], step):
            for x in range(0, image.shape[1], step):
                yield (x, y, image[y:y+window[1], x:x+window[0]])
    
    def detect(self, image=cv2.imread('function_output.png'), save=False):
        """Uses sliding window to detect baseball in image.

        Args:
            image: image to run object detection on
            save: saves area of image detected if True
            
        Baseball detection based on classifier probability.
        """

        resized = cv2.resize(image, (int(image.shape[1]), int(image.shape[0])))

        highest_p = 0
        df = pd.DataFrame(columns=[*range((resized.shape[1]-300)//32)])
        row = []
        append_flag = False
        for i, (x, y, window) in enumerate(self.sliding_window(resized)):

            if (x + 300 > resized.shape[1]) or (y + 300 > resized.shape[0]):
                if append_flag:
                    df.loc[i] = row[:-1]
                append_flag = False
                row = []
                continue

            append_flag = True
            im = Image.fromarray(window*255)
            transformed = self.transform(im)
            normalized = transformed.float().unsqueeze(0)
            output = self.net(normalized)

            prob = torch.softmax(output.data, 1)
            top_p, top_class = prob.topk(1, 1)

            if top_p > highest_p:
                highest_p = top_p
                top_window = ((x, y), (x+300, y+300))
            row.append(top_p.numpy()[0][0])
    
        # clear lower prob numbers to skew those areas down
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

        classify_window = resized[high_xy[1]:high_xy[1]+428, high_xy[0]:high_xy[0]+428]
        if save:
            cv2.imwrite('test_image.png', classify_window)
        




