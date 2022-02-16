from cgi import test
from PIL import Image, ImageDraw, ImageFilter
import os
import random

# set which image set is being generated
test_or_train = 'train'

# list of grip class names
grip_list = ["changeup", "curveball", "fastball"]

# iterate through grips and paste onto background image
for grip in grip_list:

    # grip image
    grip_img = Image.open(f"images/grips/{grip}_grip.png")
    grip_img = grip_img.resize((int(grip_img.size[0]*1.5), int(grip_img.size[1]*1.5)))
    for file in os.listdir(f"images/backgrounds/{test_or_train}"):

        # background image
        bg_img = Image.open(f"images/backgrounds/{test_or_train}/{file}")

        x_limit = bg_img.size[0] - grip_img.size[0]
        y_limit = bg_img.size[1] - grip_img.size[1]

        if x_limit < 0:
            x_limit = 0

        if y_limit < 0:
            y_limit = 0

        print(f"GRIP: {grip_img.size}")
        print(f"BG: {bg_img.size}")

        # paste grip onto background and save
        bg_img.paste(grip_img, (random.randint(0,x_limit), random.randint(0,y_limit)))
        bg_img.save(f'images/grips/augmented/{test_or_train}/{grip}/{grip}_{file}.jpg')
    print(f'Finished saving {grip} images.')
    print()

print('All images augmented and saved successfully.')





