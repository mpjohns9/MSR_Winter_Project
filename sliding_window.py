import cv2
import time

def sliding_window(image, window=(150, 150), step=32):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield (x, y, image[y:y+window[1], x:x+window[0]])


image = cv2.imread('images/grips/augmented/test/curveball/curveball_487894806.jpg-0.jpg')

resized = cv2.resize(image, (int(image.shape[1]*2), int(image.shape[0]*2)))

for (x, y, window) in sliding_window(resized):
    if (x + 150 > resized.shape[1]) or (y + 150 > resized.shape[0]):
        continue
    cv2.rectangle(resized, (x, y), (x+150, y+150), (255, 0, 0), 3)
    # cv2.rectangle(resized, (x, y), (x + window[0], y + window[1]), (255, 0, 0), 3)
    cv2.imshow("test", resized)
    cv2.waitKey(1)
    time.sleep(0.05)




