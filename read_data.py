# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:28:35 2018

@author: murata
"""

import csv, re
import pandas as pd
import numpy as np
# read muoncounter


def row_to_numpy(row, width=29, height=29):
    image = np.zeros((height*width))
    count=0
    for component in row:
        if re.search('(?<=\[\[)\d+', component):
            image[count] = re.search('(?<=\[\[)\d+', component).group(0)
        elif re.search('(?<=\[)\d+', component):
            image[count] = re.search('(?<=\[)\d+', component).group(0)
        elif re.search('(?<=\])\d+', component[::-1]):
            image[count] = re.search('(?<=\])\d+', component[::-1]).group(0)
        elif re.search('(?<=\]\])\d+', component[::-1]):
            image[count] = re.search('(?<=\]\])\d+', component[::-1]).group(0)
        else:
            image[count] = int(component)
        count+=1
    assert count==height*width, "Dimension does not match!!"
    image = image.reshape((height, width))
    return image

def csv_to_numpy(path_to_csv = "../data/1-6.2017.csv",
#                 path_to_numpy = "../data/1-6.2017.npy",
                 width=29, height=29, if_save=False):
    path_to_numpy = path_to_csv[:-3]+"npy"
    muoncount_csv = open(path_to_csv, 'r')
    reader = csv.reader(muoncount_csv)
    frame_num = len(pd.read_csv(path_to_csv, header=None))
    count = 0
    images = np.zeros((frame_num, height, width), dtype=int)
    for row in reader:
        images[count] = row_to_numpy(row)
        count += 1
    if if_save:
        np.save(path_to_numpy, images)
    return images


path_to_csv = "../data/1-6.2017.csv"
images = csv_to_numpy(path_to_csv=path_to_csv,
                      if_save=True)

"""
path_to_numpy = "../data/1-6.2017.npy"
images = np.load(path_to_numpy)
print(images.shape)
print(images[:1].sum())

path_to_csv = "../data/1-6.2017.csv"
muoncount_csv = open(path_to_csv, 'r')
reader = csv.reader(muoncount_csv)
#reader = csv.reader(muoncount_csv, lineterminator='\n')

count=0
#row0=[]
for row in reader:
    if count > 10:
        break
    else:
#        row0.append(row[0][-1])
#        print(row)
#        image = row_to_numpy(row)
        row_int = []
        for component in row:
            if re.search('(?<=\[\[)\d+', component):
                row_int.append(re.search('(?<=\[\[)\d+', component).group(0))
            elif re.search('(?<=\[)\d+', component):
                row_int.append(re.search('(?<=\[)\d+', component).group(0))
            elif re.search('(?<=\])\d+', component[::-1]):
                row_int.append(re.search('(?<=\])\d+', component[::-1]).group(0))
            elif re.search('(?<=\]\])\d+', component[::-1]):
                row_int.append(re.search('(?<=\]\])\d+', component[::-1]).group(0))
            else:
                row_int.append(int(component))
#        print(image.sum())
        print(images[count].sum())
        print(sum(map(int, row_int)))
        count+=1
#print(image)
#print(image.sum())
#print(sum(map(int, row0)))
#print(len(row0), set(row0 ))
#print(row0)
#    count+=1
"""
