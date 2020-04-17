import csv
import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle

import pandas as pd
import torchvision.transforms as transforms
# emotion_categories = ['happiness','disgust','repression','surprise','fear','others','sadness']



class casme_dataset(data.Dataset):
    def __init__(self, root, split, inp_name,max_length=141, data_list=None, sequences=None, dic=None, adj=None, transform=None):
        self.root = root
        #self.dir_path = os.path.join(root, 'Cropped') #'CASME/Cropped'
        self.dir_path = root
        self.split = split #(train/test mode)
        self.max_length = max_length
        self.dic = dic
        self.sequences = sequences
        self.classes = [i for i in self.dic.keys()]
        self.transform = transform
        self.adj = adj  ###
        
        self.inp_name = inp_name
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
            #print("casmeinp:",self.inp.shape)
        print('[dataset] casme  split=%s number of classes=%d  number of sequences=%d' % (split, len(self.classes), len(self.sequences)))

    def __getitem__(self, index):
        path, label = self.sequences[index]
        MAX_LENGTH = self.max_length  # dataset longest length in the dataset
        seq_length = len(os.listdir(path))
        front_padding = int(np.ceil((MAX_LENGTH - seq_length)/2) -1)

        #for one sequence, read all images
        temp_name = []
        
        seq_data = torch.zeros([141,3,224,224]) #[141,3,112,112]
        for img_name in os.listdir(path):
            if img_name.endswith(".jpg"):
                temp_name.append(img_name)
        temp_name = sorted(temp_name)
        seq_data = torch.zeros([len(temp_name),3,224,224]) #112,112
        
        for index,img_name in enumerate(temp_name):
            normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                            std = [0.22803, 0.22145, 0.216989])  #standard of resnet 3d
            transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            #normalize
                                            ])
            with Image.open(os.path.join(path,img_name)) as img:
                if self.transform is None:
                    img = transform(img)
                else:
                    img = self.transform(img)
                
                
                seq_data[index]=img
        seq_data = seq_data.permute(1,0,2,3) ############################################
        


        
        label = torch.tensor(label,dtype = torch.long)
        
        #au_label = torch.from_numpy(au_label) 
        #print("================================:", au_label)

        return (seq_data, path, self.inp), label
    def __len__(self):
        return len(self.sequences)
