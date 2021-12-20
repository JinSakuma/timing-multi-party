import numpy as np
import math
import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertModel, BertForSequenceClassification
import MeCab
import gensim


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos):
        x = x + self.pe[pos, :]
        return self.dropout(x)


class CTR(nn.Module):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode,
                 device,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=64,
                 hidden_size=64
                 ):
        super(CTR, self).__init__()
        self.mode = mode
        self.device = device

        self.input_size = input_size
        self.input_img_size = input_img_size
        self.input_p_size = input_p_size

        # 音響LSTM
        if self.mode in [0, 3, 4, 6]:
            self.lstm_vad = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

        # 画像LSTM
        if self.mode in [1, 3, 5, 6]:
            self.lstm_img = torch.nn.LSTM(
                input_size=input_img_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

        # 言語LSTM
        if self.mode in [2, 4, 5, 6]:
            self.lstm_l = torch.nn.LSTM(
                input_size=self.input_p_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

        if self.mode == 0 or self.mode == 2:
            self.fc = nn.Linear(hidden_size*2, 1)
        elif self.mode == 1:
            self.fc = nn.Linear(hidden_size, 1)
        elif self.mode == 3 or self.mode == 5:
            self.fc = nn.Linear(hidden_size*3, 1)
        elif self.mode == 4:
            self.fc = nn.Linear(hidden_size*4, 1)
        else:
            self.fc = nn.Linear(hidden_size*5, 1)

        if self.mode in [2, 4, 5, 6]:
            self.prev_hpa = torch.zeros(1, 1, 64).to(self.device)
            self.prev_hpb = torch.zeros(1, 1, 64).to(self.device)
            self.PAD = -1

        self.hidden_size = hidden_size
        self.hidden = None
        self.hiddenA = None
        self.hiddenB = None
        self.hiddenPA = None
        self.hiddenPB = None
        self.hidden_img = None
        
        self.text_listA = []
        self.text_listB = []
        self.scnt = 0

    def calc_voice(self, xA, xB):
        hA, self.hiddenA = self.lstm_vad(xA, self.hiddenA)
        hB, self.hiddenB = self.lstm_vad(xB, self.hiddenB)
        return hA, hB

    def calc_img(self, img):
        hImg, self.hidden_img = self.lstm_img(img, self.hidden_img)
        return hImg

    def calc_lang(self, PA, PB):
        hpA_list = []
        hpB_list = []
        for i in range(min(len(PA), len(PB))):
            pa = PA[i]
            pb = PB[i]
            if len(pa) == 0:
                hpA = self.prev_hpa
            else:
                if len(pa.shape) == 1:
                    pa = pa.reshape(1, -1)
                pa = torch.FloatTensor(pa).to(self.device)
                pa = pa.unsqueeze(0)
                hpA, self.hiddenPA = self.lstm_l(pa, self.hiddenPA)
                hpA = hpA[:, -1, :]
            hpA_list.append(hpA)

            if len(pb) == 0:
                hpB = self.prev_hpb
            else:
                if len(pb.shape) == 1:
                    pb = pb.reshape(1, -1)
                pb = torch.FloatTensor(pb).to(self.device)
                pb = pb.unsqueeze(0)
                hpB, self.hiddenPB = self.lstm_l(pb, self.hiddenPB)
                hpB = hpB[:, -1, :]
            hpB_list.append(hpB)

            self.prev_hpa = hpA
            self.prev_hpb = hpB

        hpa = torch.cat(hpA_list)
        hpb = torch.cat(hpB_list)
        return hpa.unsqueeze(0), hpb.unsqueeze(0)

    def calc_alpha(self, h):
        alpha = torch.sigmoid(self.fc(h))
        return alpha
    
    def calc_z(self, h):
        z = self.fc(h)
        return z

    def reset_state(self):
        self.text_listA = []
        self.text_listB = []
        self.scnt = 0
        
        self.hiddenA = None
        self.hiddenB = None
        self.hiddenPA = None
        self.hiddenPB = None
        self.hidden_img = None

    def back_trancut(self):
        if self.hiddenA is not None:
            self.hiddenA = (self.hiddenA[0].detach(), self.hiddenA[1].detach())
            self.hiddenB = (self.hiddenB[0].detach(), self.hiddenB[1].detach())
        if self.hidden_img is not None:
            self.hidden_img = (self.hidden_img[0].detach(), self.hidden_img[1].detach())
        if self.hiddenPA is not None:
            self.hiddenPA = (self.hiddenPA[0].detach(), self.hiddenPA[1].detach())
        if self.hiddenPB is not None:
            self.hiddenPB = (self.hiddenPB[0].detach(), self.hiddenPB[1].detach())

        if self.mode in [2, 4, 5, 6]:
            self.prev_hpa = self.prev_hpa.detach()
            self.prev_hpb = self.prev_hpb.detach()

    def reset_lang(self):
        self.prev_hpa = torch.zeros(1, 64).to(self.device)
        self.prev_hpb = torch.zeros(1, 64).to(self.device)
        

class CTR_Julius(CTR):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode,
                 device,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=45,
                 hidden_size=64
                 ):
        super().__init__(mode, input_size, input_img_size, input_p_size, hidden_size)

        self.embedding_size = 30

        # 言語LSTM
        if self.mode in [2, 4, 5, 6]:
            # 埋め込み層
            self.embedding = nn.Embedding(self.input_p_size, self.embedding_size)
            self.lstm_lng = torch.nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

    def calc_lang(self, PA, PB):
        hpA_list = []
        hpB_list = []
        for i in range(min(len(PA), len(PB))):
            pa = PA[i]
            pb = PB[i]
            if pa == self.PAD:
                hpA = self.prev_hpa
            else:
                pA = torch.tensor(pa).to(self.device, dtype=torch.long)
                emb_pA = self.embedding(pA)
                emb_pA = emb_pA.unsqueeze(0)
                hpA, self.hiddenPA = self.lstm_lng(emb_pA, self.hiddenPA)
                hpA = hpA[:, -1, :]
            hpA_list.append(hpA)

            if pb == self.PAD:
                hpB = self.prev_hpb
            else:
                pB = torch.tensor(pb).to(self.device, dtype=torch.long)
                emb_pB = self.embedding(pB)
                emb_pB = emb_pB.unsqueeze(0)
                hpB, self.hiddenPA = self.lstm_lng(emb_pB, self.hiddenPB)
                hpB = hpB[:, -1, :]
            hpB_list.append(hpB)

            self.prev_hpa = hpA
            self.prev_hpb = hpB

        hpa = torch.cat(hpA_list)
        hpb = torch.cat(hpB_list)
        return hpa.unsqueeze(0), hpb.unsqueeze(0)
    

bert_path = "/mnt/aoni04/jsakuma/bert/cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_path)
bert_model = BertForSequenceClassification.from_pretrained(
                        bert_path,
                        num_labels = 1, # ラベル数
                        output_attentions = False, # アテンションベクトルを出力するか
                        output_hidden_states = True, # 隠れ層を出力するか
                    )
    
    
class CTR_Bert(CTR):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode,
                 device,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=768,
                 hidden_size=64
                 ):
        super().__init__(mode, device, input_size, input_img_size, input_p_size, hidden_size)
      
        # 言語LSTM
        if self.mode in [2, 4, 5, 6]:
            # 埋め込み層
#             self.bert_path = "/mnt/aoni04/jsakuma/bert/cl-tohoku/bert-base-japanese-whole-word-masking"
#             self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.bert_path)
#             self.bert_model = BertForSequenceClassification.from_pretrained(
#                         self.bert_path,
#                         num_labels = 1, # ラベル数
#                         output_attentions = False, # アテンションベクトルを出力するか
#                         output_hidden_states = True, # 隠れ層を出力するか
#                     )
            
            bert_model.to(device)
            self.fc0 = nn.Linear(input_p_size, 128)
            
            self.lstm_lng = torch.nn.LSTM(
                input_size=128,
                hidden_size=hidden_size,
                batch_first=True,
            )
            
            self.text_listA = []
            self.text_listB = []

    def calc_lang(self, PA, PB):
        hpA_list = []
        hpB_list = []
        for i in range(min(len(PA), len(PB))):
            pa = PA[i]
            pb = PB[i]
            if pa != pa:
                hpA = self.prev_hpa
            else:
                self.text_listA.append(pa)  
                pa = self.bert_embedding(self.text_listA)
                pa = self.fc0(pa)
                pa = pa.unsqueeze(0)
                hpA, self.hiddenPA = self.lstm_lng(pa, self.hiddenPA)
                hpA = hpA[:, -1, :]
            hpA_list.append(hpA)

            if pb != pb:
                hpB = self.prev_hpb
            else:
                self.text_listB.append(pb)  
                pb = self.bert_embedding(self.text_listB)
                pb = self.fc0(pb)
                pb = pb.unsqueeze(0)
                hpB, self.hiddenPB = self.lstm_lng(pb, self.hiddenPB)
                hpB = hpB[:, -1, :]
            hpB_list.append(hpB)

            self.prev_hpa = hpA
            self.prev_hpb = hpB

        hpa = torch.cat(hpA_list)
        hpb = torch.cat(hpB_list)
        return hpa.unsqueeze(0), hpb.unsqueeze(0)
    
    def bert_embedding(self, text_list):
        text = ''.join(text_list)
        tokens_list = []
        bert_tokens = tokenizer.tokenize(text)
        
        # BERTの長さの制限(512)
        if len(bert_tokens)>512:
            bert_tokens = bert_tokens[-512:]
            if '[SEP]' in bert_tokens:
                index = bert_tokens.index('[SEP]')
                bert_tokens = bert_tokens[index+1:]
                
        token_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_list.append(bert_tokens)
        tokens_tensor = torch.tensor(token_ids).unsqueeze(0).to(self.device, dtype=torch.long)
        with torch.no_grad():
            outputs = bert_model(tokens_tensor)

        vectors = outputs[1][-2]
        return vectors[:, -1, :]
    
    def insert_sep_token(self):
        # 区切りを入れる(システム発話開始時)
        if len(self.text_listA)>0:
            if self.text_listA[-1] != '[SEP]':
                self.text_listA.append('[SEP]')
                
        if len(self.text_listB)>0:
            if self.text_listB[-1] != '[SEP]':
                self.text_listB.append('[SEP]')
                
        # BERTの長さの制限(512)
        if len(self.text_listA) > 512:
            self.text_listA = self.text_listA[-512:]
            index = self.text_listA.index('[SEP]')
            self.text_listA = self.text_listA[index+1:]
            
        if len(self.text_listB) > 512:
            self.text_listB = self.text_listB[-512:]
            index = self.text_listB.index('[SEP]')
            self.text_listB = self.text_listB[index+1:]
    
#     def reset_state(self):
#         self.hiddenA = None
#         self.hiddenB = None
#         self.hiddenPA = None
#         self.hiddenPB = None
#         self.hidden_img = None
#         self.text_listA = []
#         self.text_listB = []
                
    def reset_lang(self):
        self.prev_hpa = torch.zeros(1, 64).to(self.device)
        self.prev_hpb = torch.zeros(1, 64).to(self.device)
        self.text_listA = []
        self.text_listB = []
    
    
class CTR_Word2Vec(CTR):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode,
                 device,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=300,
                 hidden_size=64
                 ):
        super().__init__(mode, device, input_size, input_img_size, input_p_size, hidden_size)
        
        self.mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
      
        # 言語LSTM
        if self.mode in [2, 4, 5, 6]:
            # 埋め込み層
            self.model_path = '/mnt/aoni04/jsakuma/word2vec/WikiEntVec/20190520/jawiki.all_vectors.300d.bin'
            self.model =gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)
            
            self.lstm_lng = torch.nn.LSTM(
                input_size=input_p_size,
                hidden_size=hidden_size,
                batch_first=True,
            )
            
    def get_wakati(self, text):
        words_list = []
        self.mecab.parse('')#文字列がGCされるのを防ぐ
        node = self.mecab.parseToNode(text)
        while node:
            word = node.surface
            if word != '':
                words_list.append(word)
            node = node.next
        return words_list

    def calc_lang(self, PA, PB):
        hpA_list = []
        hpB_list = []
        for i in range(min(len(PA), len(PB))):
            pa = PA[i]
            pb = PB[i]
            if pa != pa:
                hpA = self.prev_hpa
            else:
                words_list = self.get_wakati(pa)
                try:
                    pa = self.model[words_list]
                    pa = torch.tensor(pa).to(self.device)
                    pa = pa.unsqueeze(0)
                    hpA, self.hiddenPA = self.lstm_lng(pa, self.hiddenPA)
                    hpA = hpA[:, -1, :]
                except:
                    hpA = self.prev_hpa
                
            hpA_list.append(hpA)

            if pb != pb:
                hpB = self.prev_hpb
            else:
                words_list = self.get_wakati(pb)
                try:
                    pb = self.model[words_list]
                    pb = torch.tensor(pb).to(self.device)
                    pb = pb.unsqueeze(0)
                    hpB, self.hiddenPB = self.lstm_lng(pb, self.hiddenPB)
                    hpB = hpB[:, -1, :]
                except:
                    hpB = self.prev_hpb
                    
            hpB_list.append(hpB)

            self.prev_hpa = hpA
            self.prev_hpb = hpB

        hpa = torch.cat(hpA_list)
        hpb = torch.cat(hpB_list)
        return hpa.unsqueeze(0), hpb.unsqueeze(0)


class CTR_Multitask(CTR):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=45,
                 hidden_size=64,
                 num_cls=2):
        super().__init__(mode, input_size, input_img_size, input_p_size, hidden_size)

        self.num_cls = num_cls

        if self.mode == 0 or self.mode == 2:
            self.fc_act_1 = nn.Linear(hidden_size*2, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 1:
            # self.fc_act_1 = nn.Linear(hidden_size*3, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 3 or self.mode == 5:
            self.fc_act_1 = nn.Linear(hidden_size*3, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 4:
            self.fc_act_1 = nn.Linear(hidden_size*4, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        else:
            self.fc_act_1 = nn.Linear(hidden_size*5, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)

    def predict_action(self, h):

        if self.mode != 1:
            cls = self.fc_act_1(h)
            cls = self.fc_act_2(cls)
        else:
            cls = self.fc_act_2(h)

        return cls


class CTR_Multitask_Julius(CTR_Julius):
    """
    CTR
    応答速度 alpha(t) を推定
    """
    def __init__(self,
                 mode=0,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=45,
                 hidden_size=64,
                 num_cls=2
                 ):
        super().__init__(mode, input_size, input_img_size, input_p_size, hidden_size)

        self.num_cls = num_cls

        if self.mode == 0 or self.mode == 2:
            self.fc_act_1 = nn.Linear(hidden_size*2, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 1:
            # self.fc_act_1 = nn.Linear(hidden_size*3, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 3 or self.mode == 5:
            self.fc_act_1 = nn.Linear(hidden_size*3, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        elif self.mode == 4:
            self.fc_act_1 = nn.Linear(hidden_size*4, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)
        else:
            self.fc_act_1 = nn.Linear(hidden_size*5, hidden_size)
            self.fc_act_2 = nn.Linear(hidden_size, self.num_cls)

    def predict_action(self, h):

        if self.mode != 1:
            cls = self.fc_act_1(h)
            cls = self.fc_act_2(cls)
        else:
            cls = self.fc_act_2(h)

        return cls
    

def build_CTR(args, device, input_size, input_img_size, input_p_size, hidden_size):
    if args.lang == 'ctc':
         ctr = CTR(args.mode, device, input_size, input_img_size, input_p_size, hidden_size)
    elif args.lang == 'julius':
        ctr = CTR_Julius(args.mode, device, input_size, input_img_size, input_p_size, hidden_size)
    elif args.lang == 'bert':
        ctr = CTR_Bert(args.mode, device, input_size, input_img_size, input_p_size, hidden_size)
    elif args.lang == 'word2vec':
        ctr = CTR_Word2Vec(args.mode, device, input_size, input_img_size, input_p_size, hidden_size)
    else:
        raise NotImplemented
        
    return ctr
