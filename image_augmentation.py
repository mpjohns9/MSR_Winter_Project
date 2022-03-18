"""Script to augment image dataset.

Creates larger synthetic dataset from a smaller subset of images.
In this case, baseball grips are added to a background.
"""

from PIL import Image, ImageEnhance
import os
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# set which image set is being generated
test_or_train = 'train'

# list of grip class names
grip_list = ["changeup", "curveball", "fastball"]

# grip directory
grip_source = 'cropped'
num_grips = 10

grayscale = False

resize = False

use_bg = False

name_list = [
    'marshall',
    'marshall2',
    'marshall3',
    'marshall4',
    'marshall5',
    'marshall6',
    'marshall7',
    'marshall8',
    'marshall9',
    'andru',
    'anna',
    'devesh'
    ]

# iterate through grips and paste onto background image
for name in name_list:
    for grip in grip_list:

        grip_img_list = []

        for i in range(300):
            grip_num = random.randint(0, num_grips-1)

            # grip image
            grip_img = Image.open(f"images/grips/{grip_source}/{name}_{grip}_{grip_num}.png")

            if resize:
                resize_pct = random.uniform(0.5, 0.9)
            else:
                resize_pct = 1.0

            grip_x = int(grip_img.size[0]*resize_pct)
            grip_y = int(grip_img.size[1]*resize_pct)

            grip_resized = grip_img.copy().resize((grip_x, grip_y))

            if random.randint(1,4) == 1:
                grip_resized = grip_resized.copy().transpose(Image.FLIP_LEFT_RIGHT)

            if random.randint(1,4) == 1:

                rotation = random.randint(-30, 30)
                if rotation < 0:
                    rotation = 360 + rotation

                grip_resized = grip_resized.copy().rotate(rotation)

            if not use_bg:
                filter = ImageEnhance.Brightness(grip_resized)
                grip_final = filter.enhance(random.uniform(0.75, 1.25))
                grip_final.save(f"images/grips/{test_or_train}/{grip}/{name}_{grip}-{i+1}.jpg")
                
            grip_img_list.append(grip_resized)

        if use_bg:
            bb_dict = {}
            for i in range(3):
                count = 1
                for file in os.listdir(f"images/backgrounds/{test_or_train}"):

                    grip_img = grip_img_list[random.randint(0,999)]
                    
                    # background image
                    bg_img = Image.open(f"images/backgrounds/{test_or_train}/{file}")

                    x_limit = bg_img.size[0] - grip_img.size[0]
                    y_limit = bg_img.size[1] - grip_img.size[1]

                    if x_limit < 0:
                        x_limit = 0

                    if y_limit < 0:
                        y_limit = 0

                    x_paste = random.randint(0,x_limit)
                    y_paste = random.randint(0,y_limit)
                    bg_img.paste(grip_img, (x_paste, y_paste), grip_img)

                    filter = ImageEnhance.Brightness(bg_img)
                    bg_img = filter.enhance(random.uniform(0.75, 1.25))

                    bg_img.save(f'images/grips/augmented/{test_or_train}/{grip}/{grip}_{file}-{i}.jpg')

                    key = f"{grip}_{file}-{i}.jpg"
                    bb_dict[key] = (
                        x_paste, 
                        y_paste, 
                        x_paste + grip_img.size[0], 
                        y_paste + grip_img.size[1]
                    )

                    print(count)
                    count+=1
            print(f'Finished saving {grip} images.')
            print()
        

print('All images augmented and saved successfully.')





