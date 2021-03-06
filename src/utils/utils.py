"""
-object
データセットを構築するファイル

-detail
setup  ファイルの読み込み
       feature/*csv         wizard の操作ログや VAD ラベルなど
       img_middle64/*npy    画像特徴量の中間出力 [64dim]
       spec/*npy   音響特徴量の中間出力 [256dim × 2 users]
"""
import numpy as np
import pandas as pd
import torch
from torch.utils import data


def set_random_seeds(seed):
    """Set random seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_time_index(utter):
    start_list = np.where((utter - np.array(([0]+list(utter))[:-1]))==1)[0]
    end_list = np.where((utter - np.array(([0]+list(utter))[:-1]))==-1)[0]-1
    return start_list, end_list


def get_length(st, start_list, end_list,):
    for i in range(len(start_list)):
        if i < len(start_list)-1:
            if start_list[i]<=st and end_list[i]>=st:
                length = st - end_list[i]
            elif end_list[i] < st and start_list[i+1] > st:
                length = st - end_list[i]
            elif start_list[i]>st:
                break
            else:
                pass
        else:
            if start_list[i]<=st and end_list[i]>=st:
                length = st - end_list[i]
            elif end_list[i]<st:
                length = st - end_list[i]
            else:
                pass
    
    return length


def remove_outliers(utterA, utterB, utterU, min_l=0.15, max_l=2.0, fr=0.05):
    start_listA, end_listA = get_time_index(utterA)
    start_listB, end_listB = get_time_index(utterB)
    start_listU, end_listU = get_time_index(utterU)
    
    start_list = np.asarray(start_listA.tolist()+start_listB.tolist())
    end_list = np.asarray(end_listA.tolist()+end_listB.tolist())

    idx = np.argsort(start_list)
    start_list_sorted = start_list[idx]
    end_list_sorted = end_list[idx]
    
    out_dict = {'sid':[], 'eid':[], 'length': []}
    for i in range(len(start_listU[1:])):
        st_u = start_listU[1:][i-1]
        ed_u = end_listU[1:][i-1]
        length = get_length(st_u, start_list_sorted, end_list_sorted)

        out_dict['sid'].append(st_u)
        out_dict['eid'].append(ed_u)
        out_dict['length'].append(length*fr)

    df_out = pd.DataFrame(out_dict)
    df_drop = df_out[(df_out['length']<min_l) | (df_out['length']>max_l)]
    
    return df_drop['sid'].tolist(), df_drop['eid'].tolist()


def np_to_dataframe(np_list) -> pd.DataFrame:
    """
    np -> pd.DataFrame
    """
    if type(np_list) == str:
        np_list = np.load(np_list)
        np_list = np_list[:len(np_list)//2*2]  # 奇数個なら偶数個に
#         np_list = np_list.reshape(-1,256) #20fps > 10fps
        return pd.DataFrame(np_list)
    else:  # np.load 済みなら
        return pd.DataFrame(np_list)


def add_active(df, file_name):
    """
    ELAN で 得た追加アノテーションラベルを抽出 (Active Label)
    df : feature_file に 上書きして返す
    """
    file_name = file_name.replace('/feature_50_p', '/TEXT', 1).replace('.feature.csv', '.vad.txt', 1)
    try:
        count = 0
        with open(file_name) as f:
            lines = f.readlines()
            for line in lines:
                act, _, start, end, _ = line.rstrip().split('\t')
                start = int(start) // 50
                end = int(end) // 50
                if act != 'default':
                    break

                if 'Active' in df['action'].values[start:end]:
                    count += 1
                    continue
                df['action'].values[start] = 'Active'

    except FileNotFoundError:
        print(file_name)

    return df


def extract_label(path):
    df = pd.read_csv(path)
    vad = pd.read_csv(path.replace('feature_50_p', 'vad50_safia_csv').replace('feature', 'vad'))
    df = df.drop(['utter_A', 'utter_B'], axis=1)
    df = pd.concat([df, vad], axis=1)
    df = df.fillna(0)
    return df


def get_phoneme_feature(df, phoneme_feature, phoneme_label):
    count = 0
    phoneme = []
    for k in range(len(df)):
        p = []
        if k % 2 == 0:
            for j in range(2):
                if count < len(phoneme_label):
                    if phoneme_label[count] == 1:
                        if len(p) == 0:
                            p = phoneme_feature[count]
                        else:
                            p = np.vstack([p, phoneme_feature[count]])
                    count += 1
        else:
            for j in range(3):
                if count < len(phoneme_label):
                    if phoneme_label[count] == 1:
                        if len(p) == 0:
                            p = phoneme_feature[count]
                        else:
                            p = np.vstack([p, phoneme_feature[count]])
                    count += 1
        phoneme.append(p)
    return phoneme


def hang_over(y, flag=True):
    """
    u の末端 200 ms を １ にする
    """
    if flag:
        for i in range(len(y)-1):
            if y[i] == 0 and y[i+1] == 1:
                y[i-2:i+1] = 1.
    return y


def u_t_maxcut(u, u_pred, max_frame=60):
    """
    u(t)=1 に最大値を設ける
    """
    count = 0
    for i in range(len(u)):
        if u[i] != 1:
            count = 0
        else:
            count += 1
            if count > max_frame:
                u[i] = 0
                u_pred[i] = 0
    return u, u_pred


def make_target(df):
    """
    systemno顔向きを encode
    A......0
    B......1
    A -> B, B -> A ... 2 (人の顔が映っていない)
    これらを one-hot で エンコード

    return target [Aをみている, Bをみている, どちらも見ていない]
    """
    a = df['target'].map(lambda x: 0 if x == 'A' else 1).values
    index = df['action_detail'] == 'look'
    a[index] = 1
    index = np.where(a == 1)[0]
    for i in index:
        a[i:i+3] = 1.

    return np.identity(3)[a]


keys = [
    'nil', 'N', 'N:', 'a', 'a:', 'b', 'by', 'ch', 'd', 'dy', 'e', 'e:', 'f',
    'g', 'gy', 'h', 'hy', 'i', 'i:', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o',
    'o:', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 'sp', 't', 'ts', 'ty', 'u',
    'u:', 'w', 'y', 'z', 'zy'
]
values = [i for i in range(len(keys))]
p_dict = dict(zip(keys, values))


def get_phoneme_id(phoneme_list):
    List = []
    for phonemes in phoneme_list:
        if type(phonemes) != str:
            List.append(-1)
        else:
            p_list = [i for i in phonemes.split('*') if i != '']
            id_list = []
            for i, p in enumerate(p_list):
                id_list.append(p_dict[p])

            List.append(id_list)

    out = np.asarray(List)
    if len(out.shape) > 1:
        out = out.reshape(-1)
    return out


def make_pack(pack_list, X, idxs, name):
    pre = 0
    for i, idx in enumerate(idxs):
        pack_list[i][name] = X[pre:idx]
        pre = idx

    pack_list[i+1][name] = X[pre:]
    return pack_list


# def make_pack(pack_list, X, idxs, name):
#     pre = 0
#     for i, idx in enumerate(idxs):
#         pack_list[i][name] = X[pre:idx+1]
#         pre = idx+1

#     pack_list[i+1][name] = X[pre:]
#     return pack_list


# class MyDataLoader(data.Dataset):
#     def __init__(self, pack, shuffle=True, batch_size=1):
#         super().__init__()

#         self.pack = pack
#         self.batch_size = batch_size
#         self.order = []
#         if shuffle:
#             self.order = np.random.permutation(len(self.pack))
#         else:
#             self.order = np.arange(len(self.pack))
#         self.curr_idx = -1

#     def __len__(self):
#         return int(len(self.pack)/self.batch_size)

#     def __getitem__(self, idx):

#         jdx = self.order[idx*self.batch_size:(idx+1)*self.batch_size]

#         batch = []
#         for i in range(self.batch_size):
#             batch.append(self.pack[jdx[i]])

#         return batch

#     def next(self):
#         self.curr_idx += 1
#         if self.curr_idx >= self.__len__():
#             self.curr_idx = 0
#         return self.__getitem__(self.curr_idx)

#     def on_epoch_end(self):
#         self.order = np.random.permutation(len(self.pack))

        
class MyDataLoader(data.Dataset):
    def __init__(self, data, shuffle=True, batch_size=1):
        super().__init__()

        self.data = data
        self.batch_size = batch_size
        self.order = []
        if shuffle:
            self.order = np.random.permutation(len(self.data))
        else:
            self.order = np.arange(len(self.data))
        self.curr_idx = -1

    def __len__(self):
        return int(len(self.data)/self.batch_size)

    def __getitem__(self, idx):

        jdx = self.order[idx*self.batch_size:(idx+1)*self.batch_size]

        batch = []
        for i in range(self.batch_size):
            batch.append(self.data[jdx[i]])

        return batch

    def next(self):
        self.curr_idx += 1
        if self.curr_idx >= self.__len__():
            self.curr_idx = 0
        return self.__getitem__(self.curr_idx)

    def on_epoch_end(self):
        self.order = np.random.permutation(len(self.data))

