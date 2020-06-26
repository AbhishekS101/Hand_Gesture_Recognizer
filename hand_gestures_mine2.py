# Imports
import numpy as np
import cv2
import math
from keras.models import model_from_json
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image

model = model_from_json(open("hanges.json","r").read())
model.load_weights('hange.h5')

def predt(model, img):
    data = np.asarray(img, dtype="int32")

    pprob = model.predict(data)[0]
    pclass = list(pprob).index(max(pprob))
    x = (chr(pclass + 65))
    return max(pprob), pclass, x



# Open Camera
capture = cv2.VideoCapture(0)


def main():
    while capture.isOpened():

        # Capture frames from the camera
        ret, frame = capture.read()

        # Get hand data from the rectangle sub window
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        crop_image = frame[100:300, 100:300]

        # Change color-space from BGR -> HSV
        gry = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)



        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gry, (15, 15), 0)

        image2 = cv2.resize(blur, (28, 28), interpolation=cv2.INTER_AREA)

        image3 = np.resize(image2, (28, 28, 1))
        image4 = np.expand_dims(image3, axis=0)

        pprob, pclass, des = predt(model, image4)

        cv2.putText(frame, des, (50, 50), cv2.FONT_HERSHEY_COMPLEX,2, (0, 0, 255), 2)

        # cv2.imshow("Image4",resized_img)
        cv2.imshow("Image3", frame)

        # Close the camera if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
if __name__ == '__main__':
    main()

capture.release()
cv2.destroyAllWindows()
