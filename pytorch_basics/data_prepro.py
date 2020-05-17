import os
import cv2
import numpy as np
from tqdm import tqdm   #for a progress bar UI

REBUILD_DATA = True

class DogsVSCats():
  IMG_SIZE = 50 #make images 50*50. To make the input uniform.
  CATS = "C:/Users/Vibhav/Desktop/Embedded_Systems/Year1_Q4/CV_DL/Project/pytorch_basics/kagglecatsanddogs_3367a/PetImages/Cat"
  DOGS = "C:/Users/Vibhav/Desktop/Embedded_Systems/Year1_Q4/CV_DL/Project/pytorch_basics/kagglecatsanddogs_3367a/PetImages/Dog"
  TESTING = "PetImages/Testing"
  LABELS = {CATS: 0, DOGS: 1}
  training_data = []    #to be populated with dogs and cats with labels
  catcount = 0
  dogcount = 0

  def make_training_data(self):
        for label in self.LABELS:           #iterating over the directories
          print(label)
          for f in tqdm(os.listdir(label)):  #iterating over the images in the directory
            try:
              path = os.path.join(label,f)
              img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  #in order to reduce the channels of the input.
              img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
              self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 
              #print(np.eye(2)[self.LABELS[label]])

              if label == self.CATS:      #Need to make images in the class equal
                self.catcount += 1
              elif label == self.DOGS:
                self.dogcount += 1
            except Exception as e:
              pass            
              #print(str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Cats:',self.catcount)
        print('Dogs:',self.dogcount)

if REBUILD_DATA:
  dogsvcats = DogsVSCats()
  dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))              

