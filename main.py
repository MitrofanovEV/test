import torch
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import time
IMG_SIZE = 64

transforms1 = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE//2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 6, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

id = Net().cuda()

id.load_state_dict(torch.load('id_008'))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
          (255, 255, 255), (128, 128, 128)]

print(model.names)
videofile = 'cam2.avi'
videofile1 = 'cam1.avi'
cap = cv2.VideoCapture(videofile)
cap1 = cv2.VideoCapture(videofile1)
persons = list()
ii = 0
while cap.isOpened():
    x = time.time()
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if ret and ret1:
        res = model([frame, frame1])
        pred = res.xyxy[0]
        pred1 = res.xyxy[1]
        all_ind = list()
        for bb in pred:
            if bb[5] > 0:
                continue
            img = frame[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2])]
            ii += 1
            img = Image.fromarray(img)
            img = np.array(transforms1(img))
            batch = list()
            for i, person in enumerate(persons):
                batch.append(torch.Tensor(np.concatenate((img,person))).cuda())
            if (len(batch)>0):
                out = torch.sigmoid(id(torch.stack(batch))).cpu().detach().numpy()
                ind = np.argmax(out)
                print(ind)
                value = out[ind]
            else: 
                value = 0
            if value >0.3:
                person_id = ind
            else:
                person_id = len(persons)
                persons.append(img)
            all_ind.append(person_id)
        k = 0
        for bb in pred:
            if bb[5] > 0:
                continue
            try:
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), colors[all_ind[k]%8], 2)
                cv2.putText(img=frame, text=str(all_ind[k]), org=(int(bb[0]),int(bb[1])-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                        color=(0, 255, 255), thickness=2)
            except:
                pass
            k +=1
        all_ind = list()
        for bb in pred1:
            if bb[5] > 0:
                continue
            img = frame1[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
            ii+=1
            img = Image.fromarray(img)
            img = np.array(transforms1(img))
            batch = list()
            for i, person in enumerate(persons):
                batch.append(torch.Tensor(np.concatenate((img, person))).cuda())
            if (len(batch) > 0):
                out = torch.sigmoid(id(torch.stack(batch))).cpu().detach().numpy()
                ind = np.argmax(out)
                print(ind)
                value = out[ind]
            else:
                value = 0
            print(value)
            if value > 0.3:
                person_id = ind
            else:
                person_id = len(persons)
                persons.append(img)
            all_ind.append(person_id)

        k = 0
        for bb in pred1:
            if bb[5] > 0:
                continue
            try:
                cv2.rectangle(frame1, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), colors[all_ind[k]%8], 2)
                cv2.putText(img=frame1, text=str(all_ind[k]), org=(int(bb[0]), int(bb[1]) - 10),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                            color=(0, 255, 255), thickness=2)
            except:
                pass
            k += 1
        x1 = time.time()
        cv2.putText(img=frame, text=f'fps: {1/(x1-x):.2f}', org=(0, 10),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                    color=(255, 255, 255), thickness=2)
        cv2.imshow('detect', frame)
        cv2.imshow('detect1', frame1)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('x'):
            break
    else:
        break


