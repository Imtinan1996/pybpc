import numpy as np
from random import shuffle
from collections import Counter
from PIL import ImageGrab
from sklearn.model_selection import train_test_split
import cv2
import time
import os
import pandas as pd
import pyautogui
from screenshot import GrabScreen
from imageregressor import irmodel as Regressor
from getinputs import key_check as KeyGrab
from inputkeys import Accelerate, Decelerate, Left , Right , PressKey, ReleaseKey

vertices=np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]])

balancedDataFilename='balancedTrainingData.npy'

WIDTH=80
HEIGHT=60
LAMBDA=1e-3
EPOCHS=10
TRAIN_PERCENTAGE=0.7
MODEL_NAME='py-burnout-self-driving-car.model'

def move_forward():
    PressKey(Accelerate)
    ReleaseKey(Right)
    ReleaseKey(Left)

def move_left():
    ReleaseKey(Right)
    PressKey(Left)
    
def move_right():
    ReleaseKey(Left)
    PressKey(Right)
    
    
    
def maskOutput(keys):
    #[Accelerate,Left,Right]
    output=[0,0,0]
    
    if 'W' in keys:
        output[0]=1
    if 'A' in keys:
        output[1]=1
    if 'D' in keys:
        output[2]=1
        
    return output
    
def draw_lines(image,lines):
    try:
        for line in lines:
            coordinates=line[0]
            cv2.line(image,(coordinates[0],coordinates[1]),(coordinates[2],coordinates[3]),[255,255,255],3)
    except:
        pass

def mask_area(image):
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,[vertices],255)
    masked_img=cv2.bitwise_and(image,mask)
    return masked_img

def edge_detect(original):
    processed_img=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    processed_img=cv2.Canny(processed_img,threshold1=100,threshold2=200)
    processed_img=cv2.GaussianBlur(processed_img,(5,5),0)
    
    processed_img=mask_area(processed_img)
    
    detected_lines=cv2.HoughLinesP(processed_img,1,np.pi/180,180,np.array([]),100,5)
    draw_lines(processed_img,detected_lines)
    
    return processed_img

def gatherTrainingData(training_file="trainingData.npy"):    
       
    if os.path.isfile(training_file):
        print('File exists, loading previous data!')
        training_data = list(np.load(training_file))
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    print("Starting the training process in 5 seconds, make sure the game is running and in focus")
            
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)

    print("Begin Training")
        
    while(True):

        screen = np.array(ImageGrab.grab(bbox=(0,50,800,650)))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(80,60))
        cv2.imshow('Image',screen)
        keys=KeyGrab()
        if cv2.waitKey(25) & 0xFF == ord('q') or ' ' in keys:
            print("Stop command issued")
            cv2.destroyAllWindows()
            np.save(training_file,training_data)
            break
        outputs=maskOutput(keys)
        if keys:
            print("Keys Pressed",keys,outputs)
        training_data.append([screen,outputs])

def BalanceTrainngData(training_file="trainingData.npy"):
    
    training_data=np.load(training_file)
    print("Length of data:",len(training_data))
    df = pd.DataFrame(training_data)
    print(df.head())
    print(Counter(df[1].apply(str)))
    
    left_movement=[]
    right_movement=[]
    forward_movement=[]
    
    shuffle(training_data)
    
    for data in training_data:
        img=data[0]
        inputs=data[1]
        
        if inputs==[1,0,0]:
            forward_movement.append(data)
        elif inputs==[0,1,0]:
            left_movement.append(data)
        elif inputs==[0,0,1]:
            right_movement.append(data)
        
    splice_length=min([len(forward_movement),len(left_movement),len(right_movement)])
        
    balanced_data=forward_movement[:splice_length] + left_movement[:splice_length] + right_movement[:splice_length]
        
    shuffle(balanced_data)
    
    print(len(balanced_data))
    
    np.save(balancedDataFilename,balanced_data)

def trainModel():
    model=Regressor(WIDTH,HEIGHT,LAMBDA)
    training_data=np.load(balancedDataFilename)
    train=training_data[:-500]
    test=training_data[-500:]
    
    X=np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
    Y=[i[1] for i in train]
    
    testX=np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    testY=[i[1] for i in test]
    
    model.fit({'input':X},{'targets':Y},n_epoch=EPOCHS,validation_set=({'input':testX},{'targets':testY}),snapshot_step=500,show_metric=True,run_id=MODEL_NAME)
    
    # tensorboard --logdir=foo:C:/Users/imtinan/Desktop/selfdriving-burnout/log
    
    model.save(MODEL_NAME)
    
def testModel():
    
    print("Starting the autonomous driving process in 5 seconds, make sure the game is running and in focus")
            
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)
    
    model=Regressor(WIDTH,HEIGHT,LAMBDA)
    model.load(MODEL_NAME)
    
    speedController=0
    Accelerating=False
    
    while(True):

        screen = np.array(ImageGrab.grab(bbox=(0,50,800,650)))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(80,60))
        cv2.imshow('Image',screen)
        keys=KeyGrab()
        if cv2.waitKey(25) & 0xFF == ord('q') or ' ' in keys:
            print("Testing Stopped")
            cv2.destroyAllWindows()
            ReleaseKey(Left)
            ReleaseKey(Right)
            ReleaseKey(Accelerate)
            ReleaseKey(Decelerate)
            break
        
        prediction=model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
        press=list(np.around(prediction))
        
        print(press,prediction)
        
        if Accelerating is False and speedController%2==0:
            print("Accelerating car")
            ReleaseKey(Decelerate)
            PressKey(Accelerate)
            Accelerating=True
        
        speedController+=1
        
        if Accelerating is True and speedController%2==0:
            print("Slowing down")
            ReleaseKey(Accelerate)
            PressKey(Decelerate)
            Accelerating=False
        
        #if press == [1,0,0]:
        #    move_forward()
        if press == [0,1,0]:
            move_left()
        if press == [0,0,1]:
            move_right()
        
       
testModel()
