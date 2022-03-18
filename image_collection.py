""" Script to capture and crop images quickly. """

import cv2
from PIL import Image
import numpy as np
import random

cam = cv2.VideoCapture(0)

cv2.namedWindow("grip")

grip_list = ['fastball', 'curveball', 'changeup']

ret, frame = cam.read()

x1 = random.randint(20, frame.shape[1] - 320)
y1 = random.randint(20, frame.shape[0] - 320)
x2 = x1 + 300
y2 = y1 + 300

print("**********************")
print("INSTRUCTIONS:")
print("**********************")
print("Please use a FASTBALL grip to start.")
print()
print("Place the ball in the center of the green square, and press the spacebar to save an image.")
print("*Mark sure your entire hand + ball is in the square!*")
print()
print("You will take ten images with the same grip.")
print("There will be a notice when you are done taking ten images.")
print("At this point, it will instruct you to swtich to a new grip.")
print("Switch to the new grip and take ten more images.")
print("There are 3 grips in total: fastball, curveball, and changeup.")
print("When you are finished, please send Marshall the saved images.")
print()
print("Thank you!!!")
print()
print("**********************")
escape = False
name = 'marshall8'
for i, grip in enumerate(grip_list):
    img_counter = 0;
    while img_counter < 10:
        ret, frame = cam.read()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) , 3)
        cv2.imshow("grip", frame)
        

        if not ret:
            print("failed to grab frame")
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            escape = True
            break
        elif k%256 == 32:
            ## SPACE pressed
            cropped = frame[y1+3:y2-3, x1+3:x2-3]
            x1 = random.randint(20, frame.shape[1] - 320)
            y1 = random.randint(20, frame.shape[0] - 320)
            x2 = x1 + 300
            y2 = y1 + 300
            img_name = f"{name}_{grip}_{img_counter}.png"   
            cv2.imwrite(img_name, cropped)
            print(f"{grip} image {img_counter+1} written!")
            if i < 2:
                if img_counter < 9:
                    print(f"Please take {9-img_counter} more images before switching to a {grip_list[i+1]} grip.")
                    print()
            else:
                if img_counter < 9:
                    print(f"Please take {9-img_counter} more images and then you're done!")
                    print()

            img_counter += 1
        if escape:
            break
    if escape:
        break
    if i < 2:
        print(f"Done collecting {grip} images.")
        print()
        print(f"Please switch to a {grip_list[i+1].upper()} grip.")
    else:
        print()
        print("All done! THANKS SO MUCH FOR YOUR HELP!")
        
cam.release()

cv2.destroyAllWindows()