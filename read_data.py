# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:28:35 2018

@author: murata
"""

import csv, re, datetime
import pandas as pd
import numpy as np
# read muoncounter


def row_to_numpy(row, width=29, height=29, if_reshape=True):
    row = list(map(str, row))
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
    if if_reshape:
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

"""
def eruption_data(path_to_eruption_csv="../data/erupt.csv"):
    df_eruption = pd.read_csv(path_to_eruption_csv)
    eruption_times = df_eruption["eruption_date"].value
    datetime.datetime.strptime('2017-01-18 04:50:00', '%Y/%m/%d %H:%M:%S')
    
    for eruption_time in eruption_times:
        for observation_end_time in observation_end_times:
            if eruption_time > observation_end_times:
"""

def make_observation_csv(path_to_image_csv = "../data/1-6.2014.csv",
                         path_to_eruption_list_csv="../data/eruption_list_2014-2017.csv",
                         ):
    
    df_original = pd.read_csv(path_to_image_csv, header=None)
    df_original = df_original.dropna(axis=0, how="all")    
    df_eruption = pd.read_csv(path_to_eruption_list_csv, encoding="cp932")
    df_eruption = df_eruption.dropna(axis=0, how="all")
    time_of_eruptions_str = df_eruption["time of eruption"].values
    time_of_eruptions = ["initial"]*len(time_of_eruptions_str)
    for i in range(len(time_of_eruptions)):
        time_of_eruptions[i] = datetime.datetime.strptime(time_of_eruptions_str[i], '%Y/%m/%d %H:%M')

    columns = ["end of observation",] + ["pixel%03d" % i in range(1, 842)] + ["time to eruption",]
    df = pd.DataFrame(columns = columns)
    count_eruption = 0
    time_of_eruption = time_of_eruptions[0]
    for i in range(len(df_original)):
        time_str = df_original.iloc[i,0].split(".")[0]
        start_of_observation = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        end_of_observation = start_of_observation + datetime.timedelta(minutes=10)
        
        pixel_values = list(row_to_numpy(row=df_original.iloc[i,:].values, if_reshape=False))
        print("\r%s" % time_str)
#        sys.stdout.write("\r%s" % time_str)
#        sys.stdout.flush()
        
        while end_of_observation > time_of_eruption:
            count_eruption += 1
            assert count_eruption < len(time_of_eruptions)+1
            time_of_eruption = time_of_eruptions[count_eruption]
        time_to_eruption = time_of_eruption - end_of_observation
        
        series = pd.Series([end_of_observation,]+pixel_values+[time_to_eruption,], index=df.columns)
        df = df.append(series, ignore_index = True)
    
    return df
                
df = make_observation_csv()       
print(len(df))
#path_to_csv = "../data/1-6.2017.csv"
#images = csv_to_numpy(path_to_csv=path_to_csv,
#                      if_save=True)

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
