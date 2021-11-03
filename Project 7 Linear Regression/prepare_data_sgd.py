import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

epoch = 1000
epoch_sgd = 1000
batch_size = 16
momentum = True
img_d = ''
img_n = []
img_size = 224

def show_data(base_dir):
    global img_d
    global img_n
    img_d = base_dir
    plt.figure(figsize=(15,8))
    img_list = os.listdir(img_d + 'valid/')
    img_list = sorted(img_list, key=lambda x: x[0:6])
    
    cnt = 0
    for name in img_list[::-1]:
        if len(name) == 19:
            img = Image.open(img_d + 'valid/' + name)
            img_n.append(name)
            plt.subplot(1,6,cnt+1)
            plt.imshow(img)
            cnt += 1
            if cnt == 6:
                break

def prepare_data(sub, base_dir='DATASET/'):
    test_cnt = 500
    if sub != 'test':
        age = np.loadtxt(base_dir + sub + ".txt", delimiter=',')
    else:
        age = None

    H = np.load(base_dir + 'feature_' + sub + '.npy')

    return age, H


def evaluate(w, b, age, feature):
    pred = np.dot(feature, w) + b
    loss = np.power(pred.reshape(-1,1) - age.reshape(-1,1),2).mean()
    
    plt.figure(figsize=(15,8))
    for i in range(6):
        img = cv2.imread(img_d + 'valid/' + img_n[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.putText(img, str(int(pred[::-1][i])), (0, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(int(age[::-1][i])), (180, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        plt.subplot(1,6,i+1)
        plt.imshow(img)
    
    return loss,pred


def test(w, b, feature, filename='results.txt'):
    w = w.reshape(-1,1)
    b = b.reshape(-1,1)
    pred = np.dot(feature, w) + b
    np.savetxt(filename, pred, delimiter=',')
    return pred


def evaluate_sgd_with_hidden_layer(w, b, age, feature):
    x = np.dot(feature, w[0]) + b[0].T
    x = np.maximum(x, 0)
    x = np.dot(x, w[1]) + b[1].T
    loss = np.power(x - age,2).mean()
    
    plt.figure(figsize=(15,8))
    for i in range(6):
        img = cv2.imread(img_d + 'valid/' + img_n[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.putText(img, str(int(x[::-1][i])), (0, 25),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(int(age[::-1][i])), (180, 25),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        plt.subplot(1,6,i+1)
        plt.imshow(img)
        
    return loss


def test_sgd_with_hidden_layer(w, b, feature, filename='sgd_hidden.txt'):
    x = np.dot(feature, w[0]) + b[0].T
    x = np.maximum(x, 0)
    x = np.dot(x, w[1]) + b[1].T
    np.savetxt(filename, x, delimiter=',')
    return x
