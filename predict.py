import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from train_model import Net, Network
import torch


def predict_grip(model_path):
    """ Shows video and displays classified grip on screen.

        Args:
            model_path: path to classification network to be run in real-time
    """
    nw = Network(model_path)
    net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("grip")

    transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

    prediction_list = []
    avg_pred = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, frame = cam.read()
        x1 = int(frame.shape[1]/4)
        y1 = int(frame.shape[0]/4)
        x2 = x1 + 300
        y2 = y1 + 300
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) , 3)  

        cv2.putText(frame, 
                    nw.classes[avg_pred], 
                    (50, 50), 
                    font, 1, 
                    (255, 0, 0), 
                    2, 
                    cv2.LINE_AA)
        cv2.imshow("grip", frame)

        if not ret:
            print("failed to grab frame")
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    #     elif k%256 == 32:
            ## SPACE pressed
    #         img_name = "changeup_wc_{}.png".format(img_counter)
    #         cv2.imwrite(img_name, frame)
    #         print("{} written!".format(img_name))
    #         img_counter += 1
        cropped = frame[y1+3:y2-3, x1+3:x2-3]
        print(cropped.dtype)
        im = Image.fromarray(cropped)
    #         cv2.imwrite("cropped.png", cropped)
        transformed = transform(im)
        normalized = transformed.float().unsqueeze(0)
        output = net(normalized)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(output.data, 1)
    #     print(classes[predicted])
    #     print()
    #     describe the type of font
    #     to be used.
        prediction_list.append(int(predicted[0]))
        avg_pred = int(round(np.mean(prediction_list), 0))
        if len(prediction_list) > 20:
            prediction_list = prediction_list[-20:]

    #     Use putText() method for
    #     inserting text on video
    #     print(classes[predicted])  
    cam.release()

    cv2.destroyAllWindows()