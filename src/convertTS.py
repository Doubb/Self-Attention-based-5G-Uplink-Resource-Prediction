import numpy as np
import pandas as pd

data = pd.read_csv('dataset/Highway/20211016_1147_1207_Seoul_SGLee_LG.csv', usecols=['5G_TBS',
                                                                             '4G_TBS',
                                                                             '5G_MCS',
                                                                             '4G_MCS',
                                                                             '5G_RB',
                                                                             '4G_RB',
                                                                             '5G_RSRP',
                                                                             '4G_RSRP',
                                                                             '4G_PUSCH_POWER',
                                                                             '5G_PUSCH_POWER'])

# make aggregated TBS column (target column)
# and normalize RB values
data = data.iloc[:500000, :] # read maximum 500,000 lines because of computing power
data['AGG_TBS'] = data['5G_TBS'] + data['4G_TBS']
data['5G_RB'] = data['5G_RB'].div(275)
data['4G_RB'] = data['4G_RB'].div(100)
data.pop('5G_TBS')
data.pop('4G_TBS')
dataset_agg = data.pop('AGG_TBS')
seq_size = 50 # length of input sequence (1,000 ms)
pred_size = 5 # length of prediction (100ms)
num_features = 8
print('load dataset done')

# make 1000ms sequence dataset (t1)
t1 = data[:].iloc[0:].reset_index(0, drop=True)
for i in range(1, seq_size):
    t2 = data[:].iloc[i:].reset_index(0, drop=True)
    t1 = pd.concat([t1, t2], axis=1)

t1 = t1[t1.index % pred_size == 0]
t1 = t1.dropna(axis=0)
print('t1 done')
print(t1)

# calculate average AGG_TBS for next 100ms
df_agg = pd.DataFrame(columns=['AGG_TBS'])
for idx in range(int((dataset_agg.shape[0] - seq_size) / pred_size)):
    avg = 0
    for j in range(pred_size):
        tmp = dataset_agg.iloc[idx * pred_size + j + seq_size]
        avg += tmp
    avg = avg / 5.0
    df_agg = df_agg.append({'AGG_TBS': avg}, ignore_index=True)

# concatenate with t1
df_agg = df_agg.rename(index=lambda x: x * 5)
dataset = pd.concat([t1, df_agg], axis=1)
dataset = dataset.dropna(axis=0)
print('dataset done')
print(dataset)
print()

# make .ts files for TRAIN
text_str = ""
text_file = open("Highway_20211016_1147_1207_AGG_TBS.ts", "a")
for i in range(dataset.shape[0]):
    for j in range(num_features):
        for k in range(seq_size):
            text_str += str(round(dataset.iloc[i, j + k * num_features], 4))  # each feature value
            if k == seq_size - 1:
                break
            text_str += ","
        text_str += ":"
    text_str += str(round(dataset.iloc[i, num_features * seq_size], 4))  # agg tbs
    text_str += "\n"
    text_file.write(text_str)
    text_str = ""

    if i % 100 == 0:
        print('writing .ts file ', i)
