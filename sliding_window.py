import cv2

def sliding_window(image, model, window=(150, 150), step=8):
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            yield (x, y, image[y:y+window[1], x:x+window[0]])


image = cv2.imread('images/grips/augmented/test/curveball/curveball_487894806.jpg-0.jpg')





