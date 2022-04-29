import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from io import open
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
from sklearn.preprocessing import LabelEncoder

core = cpu_count()


def parallelize(data, func, num_of_processes=core):
    data_split = np.array_split(data, num_of_processes)
    with Pool(num_of_processes) as pool:
        data_list = pool.map(func, data_split)
    data = pd.concat(data_list)
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=core):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_fre_threshold", type=int, default=10)
    parser.add_argument("--user_fre_threshold", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='delicious_process')
    parser.add_argument("--max_session_length", type=int, default=100)


    opt = parser.parse_args()
    print(opt)

    print('Loding data...')

    data = pd.read_csv('dataset_delicious/user_taggedbookmarks-timestamps.dat', delimiter='\t')
    data['time']=pd.to_datetime(data['timestamp'], unit='ms')

    item_counts = data['tagID'].value_counts() # Specific column 
    to_remove = item_counts[item_counts <= opt.item_fre_threshold].index
    data['tagID'].replace(to_remove, np.nan, inplace=True)
    data = data.dropna()
    data['tagID'] = data['tagID'].astype(np.int64)

    user_counts = data['userID'].value_counts() # Specific column 
    to_remove = user_counts[user_counts <= opt.user_fre_threshold].index
    data['userID'].replace(to_remove, np.nan, inplace=True)
    data = data.dropna()
    data['userID'] = data['userID'].astype(np.int64)


    data['day'] = data['time'].dt.date
    data['session_id'] = data['userID'].astype(str) + '-' + data['bookmarkID'].astype(str) + '-' + data['day'].astype(str)

    data_user_set = set(data['userID'])

    contact = pd.read_csv('dataset_delicious/user_contacts-timestamps.dat', delimiter='\t')
    contact = contact[(contact['userID'].apply(lambda x: int(x) in data_user_set)) & (contact['contactID'].apply(lambda x: int(x) in data_user_set))]
    contact['timestamp']=pd.to_datetime(contact['timestamp'], unit='ms')

    user_encoder = LabelEncoder()
    data['userID'] = user_encoder.fit_transform(data['userID'])
    contact['userID'] = user_encoder.transform(contact['userID'].apply(int))
    contact['contactID'] = user_encoder.transform(contact['contactID'].apply(int))

    item_encoder = LabelEncoder()
    data['tagID'] = item_encoder.fit_transform(data['tagID'])

    user_encoder_map = pd.DataFrame(
        {'encoded': range(len(user_encoder.classes_)), 'userID': user_encoder.classes_})
    user_encoder_map.to_csv(opt.save_dir + '/user_encoder_map.csv', index=False)

    item_encoder_map = pd.DataFrame(
        {'encoded': range(len(item_encoder.classes_)), 'tagID': item_encoder.classes_})
    item_encoder_map.to_csv(opt.save_dir + '/item_encoder_map.csv', index=False)

    session_list = data['session_id'].unique()
    session_data = pd.DataFrame({'session_id':session_list})

    def get_chart(whole_frame, row):
        chart = whole_frame[whole_frame['session_id'] == row.session_id]
        chart.sort_values('time', inplace=True)
        user = chart.iloc[0, 0]
        timestamp = chart.iloc[0, 4]
        session = chart['tagID'].tolist()
        if len(session) > opt.max_session_length:
            session = session[-opt.max_session_length:]
        print(timestamp)
        return user, session, timestamp

    def session_building(session_data, data):
        pack = parallelize_on_rows(session_data, partial(get_chart, data.copy(deep=True)))
        session_data['user_encoded'] = pack.apply(lambda x: x[0])
        session_data['session'] = pack.apply(lambda x: x[1])
        session_data['timestamp'] = pack.apply(lambda x: x[2])
        return session_data

    session_data = session_building(session_data, data)

    session_data.sort_values("timestamp",inplace=True)
    contact.sort_values("timestamp",inplace=True)

    session_data.to_pickle(opt.save_dir + "/session_data.pkl")
    contact.to_pickle(opt.save_dir + "/contact.pkl")


            




