import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random
from src.utils.utils import add_active, np_to_dataframe, u_t_maxcut, make_target, get_phoneme_id
from src.utils.utils import MyDataLoader, remove_outliers, get_length, get_time_index, make_pack


def setup(PATH, file_id, dense_flag=False, elan_flag=False):
    feature_dict = {'feature': [], 'voiceA': [], 'voiceB': [], 'img': [], 'wordA':[], 'wordB': [], 'wordU': [],
                    'phonemeA': [], 'phonemeB': [], 'u': []}
    for i, idx in enumerate(tqdm(file_id)):
        feature_file = os.path.join(PATH, 'feature_50_word', '{}.feature.csv'.format(idx))
        gaze_file = os.path.join(PATH, 'img_middle64', '{}.img_middle64.npy'.format(idx))
        img_middle_feature_fileA = os.path.join(PATH, 'spec', '{}.A.npy'.format(idx))
        img_middle_feature_fileB = os.path.join(PATH, 'spec', '{}.B.npy'.format(idx))
        phoneme_middle_feature_fileA = os.path.join(PATH, 'phoneme_ctc_1112', '{}A.npy'.format(idx))
        phoneme_middle_feature_fileB = os.path.join(PATH, 'phoneme_ctc_1112', '{}B.npy'.format(idx))
        u_file = os.path.join(PATH, 'u_t_50', '{}.feature.csv'.format(idx))

        df = pd.read_csv(feature_file)
        df_u = pd.read_csv(u_file)[['u_pred', 'u_B_pred']]

        if elan_flag:
            df = add_active(df, feature_file)

        gaze = np.load(gaze_file)
        gaze_new = np.zeros([len(gaze)*2, 64])
        for j in range(len(gaze)):
            gaze_new[j*2] = gaze[j]
            gaze_new[j*2+1] = gaze[j]

        gaze = pd.DataFrame(gaze_new)
        img = np_to_dataframe(img_middle_feature_fileA)
        imgB = np_to_dataframe(img_middle_feature_fileB)
        
        wordA = df['wordA'].values
        wordB = df['wordB'].values
        wordU = df['U'].values

        phonA = np.load(phoneme_middle_feature_fileA, allow_pickle=True)
        phonB = np.load(phoneme_middle_feature_fileB, allow_pickle=True)
        min_len = min([len(df), len(df_u), len(img), len(gaze), len(phonA), len(phonB)])
        
        # vad_file ???????????????
        feature_dict['feature'].append(df[:min_len])
        feature_dict['u'].append(df_u[:min_len])
        feature_dict['voiceA'].append(img[:min_len])
        feature_dict['voiceB'].append(imgB[:min_len])
        feature_dict['img'].append(gaze[:min_len])
        feature_dict['wordA'].append(wordA[:min_len])
        feature_dict['wordB'].append(wordB[:min_len])
        feature_dict['wordU'].append(wordU[:min_len])
        feature_dict['phonemeA'].append(phonA[:min_len])
        feature_dict['phonemeB'].append(phonB[:min_len])

    return feature_dict


def preprocess(feat, target_type, model_type='rtg', phase='train', lang='ctc'):

    data = []
    threshold = 1.0
    for i in range(len(feat['feature'])):

        # u??????(VAD??????)
        utterA = feat['feature'][i]['utter_A'].values
        utterB = feat['feature'][i]['utter_B'].values
        utterU = feat['feature'][i]['utter_R'].values

        u = (1 - utterA) * (1 - utterB)  # ?????????????????? A???B???OFF?????? u=1

        # u?????????
        ua = feat['u'][i]['u_pred'].values
        ub = feat['u'][i]['u_B_pred'].values
        u_pred = np.min(np.stack([ua, ub], axis=1), axis=1)

        #???????????????
        voiceA = feat['voiceA'][i].values
        voiceB = feat['voiceB'][i].values

        # ???????????????
        if not target_type:
            target = feat['feature'][i]['target'].map(lambda x: 0 if x == 'A' else 1).values
            target = target.reshape(len(target), 1)
        else:
            target = make_target(feat['feature'][i])

        img = feat['img'][i].values
        img = np.append(target, img, axis=1)

        # ???????????????
        if lang == 'julius':
            pa = feat['feature'][i]['A_phoneme'].tolist()
            phonemeA = get_phoneme_id(pa)

            pb = feat['feature'][i]['B_phoneme'].tolist()
            phonemeB = get_phoneme_id(pb)

        else:
            phonemeA = feat['phonemeA'][i]
            phonemeB = feat['phonemeB'][i]


        wordA = feat['wordA'][i]
        wordB = feat['wordB'][i]
        wordU = feat['wordU'][i]

        # ???????????????
        y = feat['feature'][i]['action'].map(lambda x: threshold if x in ['Active','Passive'] else 0).values
        action = feat['feature'][i]['action'].map(lambda x: 1 if x == 'Passive' else 0).values
        action_c = feat['feature'][i]['action'].map(lambda x: 1 if x in ['Active-Continue','Passive-Continue'] else 0).values

        def get_offset(u_, idx):
            cnt = 0
            for uu in u_[:idx][::-1]:
                cnt += 1
                if uu == 0:
                    break

            return cnt

        r = utterU[1:]-utterU[:-1]
        start_list = np.where(r == 1)[0] + 1
        end_list = np.where(r == -1)[0]

    #     assert len(start_list)==len(end_list)-1, "nums of start and end are not matched"

        for j in range(len(end_list)-1):
            start = end_list[j]+1
            end = end_list[j+1]
            y_ = y[start:end]
            action_ = action[start:end]
            action_c_ = action_c[start:end]
            u_ = u[start:end]
            u_pred_ = u_pred[start:end]

            if len(np.where(y_==1)[0])==0: # ?????????????????????????????????????????????
                continue

            try:
                idx = np.where(y_==1)[0][0]
            except:
                print(np.where(y_==1))
                print(aaaa)

            if u_[idx] == 0: #???????????????????????????????????????
                continue

            offset = get_offset(u_, idx)
            endpoint = idx - offset

            if offset>60: #???????????????3???????????????????????????????????????
                continue

            if end-(start+endpoint)>60: #???????????????3????????????????????????
                end = start + endpoint + 60

            y_ = y[start:end]
            action_ = action[start:end]
            action_c_ = action_c[start:end]
            u_ = u[start:end]
            #u_[endpoint:] = 0
            u_pred_ = u_pred[start:end]
            
            if model_type=='rtg' or model_type=='satg':
                y_[action_c_==1] = 1 

            voiceA_ = voiceA[start:end]
            voiceB_ = voiceB[start:end]
            img_ = img[start:end]
            phonemeA_ = phonemeA[start:end]
            phonemeB_ = phonemeB[start:end]
            wordA_ = wordA[start:end]
            wordB_ = wordB[start:end]
            wordU_ = wordU[start:end]
            
            if voiceA_.shape[0]>600: # ????????????(30?????????)??????????????? (????????????????????????)
                continue
                
            uends = np.where(u_[1:]-u_[:-1] == -1)[0]
            pre=0
            entries = []
            for ue in uends:
                if y_[ue]>0 or pre >= ue-2:
                    continue
                
                ue = ue+2
                entry = {
                    "y": y_[pre:ue],
                    "action": action_[pre:ue],
                    "u": u_[pre:ue],
                    "u_pred": u_pred_[pre:ue],
                    "voiceA": voiceA_[pre:ue],
                    "voiceB": voiceB_[pre:ue],
                    "img": img_[pre:ue],
                    "phonemeA": phonemeA_[pre:ue],
                    "phonemeB": phonemeB_[pre:ue],
                    "wordA": wordA_[pre:ue],
                    "wordB": wordB_[pre:ue],
                    "wordU": wordU_[pre:ue],
                    "endpoint": endpoint,
                    "offset": offset,
                }
                pre = ue+1

                entries.append(entry)
                
            if len(y_[pre:])>1:
                entry = {
                        "y": y_[pre:],
                        "action": action_[pre:],
                        "u": u_[pre:],
                        "u_pred": u_pred_[pre:],
                        "voiceA": voiceA_[pre:],
                        "voiceB": voiceB_[pre:],
                        "img": img_[pre:],
                        "phonemeA": phonemeA_[pre:],
                        "phonemeB": phonemeB_[pre:],
                        "wordA": wordA_[pre:],
                        "wordB": wordB_[pre:],
                        "wordU": wordU_[pre:],
                        "endpoint": endpoint,
                        "offset": offset,
                    }

                entries.append(entry)
            
            if len(entries)>0:
                data.append(entries)
            
    return data
        

def get_dataloader(args):
    file_id = [file.replace('.feature.csv', '') for file in sorted(os.listdir(os.path.join(args.input, 'feature_50_word')))]
    file_id.remove('20190703150134')
    
    assert len(file_id) == 110, "file num must be 110"
    
    random.seed(0)
    random.shuffle(file_id)
    
    train_num = 90
    val_num = 10
    test_num = 10
    feat_train = setup(args.input, file_id[:train_num], dense_flag=args.dense_flag, elan_flag=args.elan_flag)
    feat_val = setup(args.input, file_id[train_num:train_num+val_num], dense_flag=args.dense_flag, elan_flag=args.elan_flag)
    feat_test = setup(args.input, file_id[-test_num:], dense_flag=args.dense_flag, elan_flag=args.elan_flag)
    
    print("preprocessing ...")
    dataset_train = preprocess(feat_train, target_type=args.target_type, model_type=args.model_type, phase='train', lang=args.lang)
    dataset_val = preprocess(feat_val, target_type=args.target_type, model_type=args.model_type, phase='val', lang=args.lang)
    dataset_test = preprocess(feat_test, target_type=args.target_type, model_type=args.model_type, phase='val', lang=args.lang)

    print("finish!")
    train_loader = MyDataLoader(dataset_train, shuffle=True, batch_size=args.batch_size)
    val_loader = MyDataLoader(dataset_val, shuffle=False, batch_size=args.batch_size)
    test_loader = MyDataLoader(dataset_test, shuffle=False, batch_size=args.batch_size)

    dataloaders_dict = {"train": train_loader, "val": val_loader, "test": test_loader}
    return dataloaders_dict
