import cv2
import torch
import torchvision.transforms as transforms
from training import ConvNet
from training2 import ConvNet_ident
import PIL


def neuro_detect():
    path = "my_model1.pt"
    net = torch.load(path)
    net.eval()
    return net

def neuro_ident():
    path = "my_model4.pt"
    net = torch.load(path)
    net.eval()
    return net


test_transforms = transforms.Compose([transforms.Resize(120),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                      ])

def main():
    classes = ['Mary', 'Mihail', 'Sergay', 'Serik', 'Yaroslav']
    net = neuro_detect()
    net_ident = neuro_ident()
    haar_file = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    while True:
        (_, im) = webcam.read()
        faces = face_cascade.detectMultiScale(
            im,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10)
        ) 
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = im[y:y + h, x:x + w]
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            face = cv2.resize(face, (120, 120))
            img = PIL.Image.fromarray(face)
            image_tensor = test_transforms(img).float()
            image_tensor = image_tensor.unsqueeze_(0)
            #img = transforms.ToPILImage()(image_tensor[0])
            #img.show()
            predict = net(image_tensor)
            predict = torch.exp(predict)
            _,pr = torch.max(predict.data, 1)
            print(pr)
            if pr==0:
                cv2.putText(im, f'no mask, {predict[0][0]}', (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                #cv2.putText(im, f'mask, {predict[0][1]}', (x - 10, y - 10),
                #            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                result = net_ident(image_tensor)
                _, predicted = torch.max(result.data, 1)
                print(classes[predicted], result[0])
                if result[0][predicted]>=5:
                    cv2.putText(im, f'{classes[predicted]} in mask, {predict[0][1]}', (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                else:
                    cv2.putText(im, f'anonymous in mask, {predict[0][1]}', (x - 10, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow('Detected mask', im)
        key = cv2.waitKey(10)
        if key == 27:
            break

if __name__ == '__main__':
    main()