# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:28:35 2018

@author: murata
"""

import csv, re, datetime, math
import pandas as pd
import numpy as np
# read muoncounter

def get_component(component):
    component = str(component)
    if re.search('(?<=\[\[)\d+', component):
        comp = re.search('(?<=\[\[)\d+', component).group(0)
    elif re.search('(?<=\[)\d+', component):
        comp = re.search('(?<=\[)\d+', component).group(0)
    elif re.search('(?<=\])\d+', component[::-1]):
        comp = re.search('(?<=\])\d+', component[::-1]).group(0)
    elif re.search('(?<=\]\])\d+', component[::-1]):
        comp = re.search('(?<=\]\])\d+', component[::-1]).group(0)
    else:
        comp = int(component)
    return comp

def get_end_time(component):
    start_time = datetime.datetime.strptime(component.split(".")[0], '%Y-%m-%d %H:%M:%S')
    end_time = start_time + datetime.timedelta(minutes=10)
    return end_time

    
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

# ミュオグラムのデータを統合
def combine_muogram(path_to_csvs=["../data/1-6.2014.csv","../data/1-6.2017.csv"],
                    if_save=False,
                    ):
    path_to_total_csv = "../data/1-6.2014-2017.csv"
    dfs = []
    for path_to_csv in path_to_csvs:
        dfs.append(pd.read_csv(path_to_csv, header=None))
    df_total = pd.concat(dfs)
    
    if if_save:
        df_total.to_csv(path_to_total_csv, header=False, index=False)
    
    return df_total

# ミュオグラフィの観測時間が１０分刻みかどうか確認
def check_timedelta(path_to_image_csv = "../data/1-6.2014-2017.csv",
                    ):
    df_image = pd.read_csv(path_to_image_csv, header=None)
    df_image = df_image.dropna(axis=0, how="all")    
    time_str = df_image.iloc[0,0].split(".")[0]
    start_of_observation = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    count_error = 0
    for i in range(1,len(df_image)):
        time_str = df_image.iloc[i,0].split(".")[0]
        time_step = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')-start_of_observation
#        print(time_step)
        if not time_step==datetime.timedelta(minutes=10):
            count_error += 1
            print(time_str, time_step)
#        print("\r{0}".format(time_step), end="")
#        assert time_step==datetime.timedelta(minutes=10), "{0},{1}".format(df_image.iloc[i-1,0].split(".")[0],time_str)
        start_of_observation = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    print(count_error)
    print("initial time =", df_image.iloc[0,0].split(".")[0])
    print("final time =", df_image.iloc[-1,0].split(".")[0])

def datetime_to_str(time):
    return time.strftime('%Y-%m-%d %H:%M:%S')

def str_to_datetime(time="",
                    slash_dash="slash",
                    ):
    if slash_dash=="slash":
        return datetime.datetime.strptime(time, '%Y/%m/%d %H:%M')
    if slash_dash=="dash":
        return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    
# ミュオグラフィのデータを扱いやすい形（時間、画素値に分ける）に整形
def reform_muogram(path_to_image_csv = "../data/1-6.2014-2017.csv",
                   ):
    path_to_reform_csv = path_to_image_csv[:-4] + "_reform.csv"
    df_image = pd.read_csv(path_to_image_csv, header=None)
    df_image = df_image.dropna(axis=0, how="all")    
    
    df_pixels = df_image.applymap(get_component)
    df_pixels.columns = ["pixel%03d" % xy for xy in range(1, 842)]
    
    df_times = df_image.loc[:,:0]
    df_times = df_times.applymap(get_end_time)
    df_times.columns = ["end of observation"]
    
    df_reform = pd.concat([df_times,df_pixels], axis=1)
    df_reform.to_csv(path_to_reform_csv, index=None)
    print(df_reform.info())

    
# 観測終了時間、画素値、噴火までの時間を csv 形式で保存
def make_observation_csv(# path_to_image_csv = "../data/1-6.2014.csv",
                         path_to_reform_csv ="../data/1-6.2014-2017_reform.csv",
                         path_to_eruption_list_csv="../data/eruption_list_2014-2017.csv",
                         path_to_observation_csv = "../data/observation.csv",
                         time_unit="hour",
                         ):
    # time to eruption の単位時間 を秒単位で表す
    if time_unit == "minute":
        t_u = 60.0     
    elif time_unit == "hour":
        t_u = 3600.0 
    elif time_unit == "day":
        t_u = 3600.0 * 24.0
    df_reform = pd.read_csv(path_to_reform_csv)
    df_reform = df_reform.dropna(axis=0, how="all")
    end_of_observations = list(df_reform["end of observation"].values)    
#    df_image = pd.read_csv(path_to_image_csv, header=None)
#    df_image = df_image.dropna(axis=0, how="all")    
    df_eruption = pd.read_csv(path_to_eruption_list_csv, encoding="cp932")
    df_eruption = df_eruption.dropna(axis=0, how="all")
    time_of_eruptions_str = df_eruption["time of eruption"].values
    time_of_eruptions = ["initial"]*len(time_of_eruptions_str)
    for i in range(len(time_of_eruptions)):
        time_of_eruptions[i] = datetime.datetime.strptime(time_of_eruptions_str[i], '%Y/%m/%d %H:%M')

#    columns = ["end of observation",] + ["pixel%03d" % xy for xy in range(1, 842)] + ["time to eruption",]
#    df = pd.DataFrame(columns = columns)
    count_eruption = 0
    time_of_eruption = time_of_eruptions[0]
    time_to_eruptions = [0]*len(end_of_observations)
    for i in range(len(end_of_observations)):
#    for i in range(5):
#        time_str = df_reform.iloc[i,0].split(".")[0]
        time_str = end_of_observations[i]
#        start_of_observation = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
#        end_of_observation = start_of_observation + datetime.timedelta(minutes=10)
        end_of_observation = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        
#        pixel_values = list(row_to_numpy(row=df_image.iloc[i,:].values, if_reshape=False))
        
        while end_of_observation > time_of_eruption:
            count_eruption += 1
#            print("\r%d, %d" % (count_eruption, len(time_of_eruptions)) )
            assert count_eruption < len(time_of_eruptions)+1
            time_of_eruption = time_of_eruptions[count_eruption]
        print("\r{0},{1}".format(end_of_observation, time_of_eruption), end="")
        end_of_observations[i] 
        time_to_eruption = time_of_eruption - end_of_observation
        time_to_eruptions[i] = time_to_eruption.total_seconds() / t_u
        
    print("")
    print(time_to_eruptions[-1])
    
    df_eruption = pd.DataFrame(time_to_eruptions,columns=["time to eruption"])

    df_observation = pd.concat([df_reform,df_eruption], axis=1)
    df_observation.to_csv(path_to_observation_csv, index=None)
    print(df_observation.info())
        
#        print(len(pixel_values), len(columns))
        
#        series = pd.Series([end_of_observation,]+pixel_values+[time_to_eruption,], index=df.columns)
#        df = df.append(series, ignore_index = True)
    
#    df.to_csv(path_to_observation_csv, index=None)
    
#    return df

# 観測時間分の観測データが存在する行だけを抽出する
def remove_time_deficit(path_to_observation_csv = "../data/observation.csv",
                        path_to_observation_hour_csv = "../data/observationhour%03d.csv",
                        observation_hour=24, # period_observation 時間の観測データを使う
                        ):
    time_step = observation_hour*6-1 # １０分単位に変換、time_step*10 分までの継続データがあればその行を残す
    df_observation = pd.read_csv(path_to_observation_csv)
    df_observation = df_observation.dropna(axis=0, how="all")    
    eoos = df_observation["end of observation"].values # eoo=end of observation
#    print(end_of_observations)
#    e_o_b_before = datetime.datetime.strptime(end_of_observations[0], '%Y-%m-%d %H:%M:%S')
#    columns = ["end of observation",] + ["pixel%03d" % xy for xy in range(1, 842)] + ["time to eruption",]
#    df = pd.DataFrame(columns = columns)
    eoo_time_step = []
    for t in range(time_step, len(df_observation)):
#        time_str = df_image.iloc[t,0].split(".")[0]
        eoo_before = datetime.datetime.strptime(eoos[t-time_step], '%Y-%m-%d %H:%M:%S')
        eoo_after = datetime.datetime.strptime(eoos[t], '%Y-%m-%d %H:%M:%S')
        time_delta = eoo_after-eoo_before
        if time_delta == datetime.timedelta(minutes=10*time_step): 
            # 時間差が行の間隔に等しい、つまり欠損データが無ければそれを加える
            eoo_time_step.append(eoo_after)
            print("\r%d" % t, end="")
    eoo_time_step = list(map(datetime_to_str, eoo_time_step))
#    print(e_o_b_time_step[:3])
    print("")
    print(len(df_observation))
    
    # df_observationから e_o_b_time_step に含まれるものだけ抽出しよう
    df_restricted = df_observation[df_observation["end of observation"].isin(eoo_time_step)]
    print(len(df_restricted))
    print(df_restricted[:3])
    df_restricted.to_csv(path_to_observation_hour_csv % observation_hour)

"""
# time to eruption を再定義
# ここは学習時でもいいかも
def deform_times(path_to_observation_hour_csv = "../data/obsevationhour%03d.csv",
                 path_to_observation_hour_prediction_hour_csv =  "../data/observationhour%03d_predictionhour%03d.csv",
                 observation_hour=24,
                 prediction_hour=24, #単位は hour
                 ):
#    path_to_observation_ts_th = path_to_observation_time_step_csv[:-4] +\
#    "timethreshold%03d.csv" % prediction_hour
    # 一定時間後を圧縮
    def deform_time(time):
        if time > prediction_hour:
            arg_tanh = (time-prediction_hour) / prediction_hour
            time = prediction_hour + prediction_hour*math.tanh(arg_tanh)
        return time
    df_observation_t_s = pd.read_csv(path_to_observation_hour_csv % observation_hour)
    time_to_eruptions = df_observation_t_s["time to eruption"].values
    deformed_times = list(map(deform_time, time_to_eruptions))
    print(len(deformed_times))
    print(deformed_times[:3])
    print(max(deformed_times))
    
    # time to eruption を置き換える
    df_observation_t_s = df_observation_t_s.iloc[:,:-1]
    df_eruption = pd.DataFrame(deformed_times,columns=["time to eruption"])
    df_deformed_time = pd.concat([df_observation_t_s,df_eruption], axis=1)
    df_deformed_time.to_csv(path_to_observation_hour_prediction_hour_csv % (observation_hour, prediction_hour), index=None)
"""    
#            series = pd.Series(df_observation.iloc[t].values, index=df_observation.columns)
#            df.append(series, ignore_index = True)
#        start_of_observation = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
#    df.to_csv(path_to_observation_time_step_csv % time_step, index=None)


def main():  
#    df = combine_muogram(if_save=True)
#    check_timedelta(path_to_image_csv = "../data/1-6.2014-2017.csv")
#    reform_muogram(path_to_image_csv = "../data/1-6.2014-2017.csv")
    make_observation_csv()
#    remove_time_deficit(observation_hour=6)         
#    deform_times(observation_hour=6, prediction_hour=24)

if __name__ == '__main__':
    main()

#path_to_csv = "../data/1-6.2017.csv"
#images = csv_to_numpy(path_to_csv=path_to_csv,
#                      if_save=True)

