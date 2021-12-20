import os
import numpy as np
import pandas as pd
import math


def timing_evaluation(y_pred, y_true, u_label, threshold=0.5, frame=50):
   
    target = False
    pred = False
    flag = True
    fp_flag = True
    tn_flag = True
    TP, FP, FN, TN = 0, 0, 0, 0
    start_frame = 0
    
    pred_frame, target_frame = -1, -1
    for i in range(1, len(y_pred)-1):
        if u_label[i]<=0:
            if y_pred[i] >= threshold and fp_flag:
                FP+=1
                fp_flag = False
                tn_flag = False
        else:
            if u_label[i-1]<=0 and u_label[i]==1:
                if tn_flag:
                    TN+=1
                tn_flag = True
            
            #  予測が閾値を超えたタイミング
            if y_pred[i] >= threshold and flag:
                pred = True
                flag = False
                pred_frame = i

            #  正解ラベルのタイミング
            if y_true[i] > 0:
                target = True
                target_frame = i

        
    flag = True
    if pred and target:
        TP += 1
    elif pred:
        FP += 1
    elif target:
        FN += 1
    else:
        TN += 1

    return TP, FP, FN, TN, pred_frame*frame, target_frame*frame


def evaluation(pred, label, uttr, thres=0.5, err=400):
    dic={"TP": 0, "TP_pred": [], "TP_label": [], "FN": 0, "FN_label": [], "FP": 0, "TN": 0}
    for i in range(len(pred)):
        TP, FP, FN, TN, predict, target = timing_evaluation(pred[i], label[i], uttr[i], threshold=thres)

        if TP>0:
            dic["TP"]+=1
            dic["TP_label"].append(target)
            dic["TP_pred"].append(predict)
        if FN>0:
            dic["FN"]+=1
            dic["FN_label"].append(target)
        if FP>0: 
            dic["FP"]+=FP
        if TN>0:
            dic["TN"]+=TN

    type_list = [1 for i in range(dic["TP"])] + [0 for i in range(dic["FN"])]
    df = pd.DataFrame({
        'type': type_list, 
        'target': dic["TP_label"]+dic["FN_label"],
        'pred': dic["TP_pred"]+[3000-dic["FN_label"][i] for i in range(dic["FN"])],
    })

    df['error'] = df['target'].values - df['pred'].values
    
    mae = np.array([abs(i) for i in df['error'].values]).mean()
    FP = dic['FP']
    TN = dic['TN']

    recall = len(df[abs(df['error'])<=err]) / len(df)
    precision = len(df[abs(df['error'])<=err]) / (FP+len(df[abs(df['error'])<=err]))
    f1 = 2 * recall * precision / (recall + precision)
    
    return precision, recall, f1, mae
