
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from src.models.SATG.transformer import TransformerModel
from src.models.SATG.elapsed_time_encoding import ElapsedTimeEncoding


class SATG(nn.Module):
    def __init__(self,
                 args,
                 device,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=128,
                 hidden_size=64,
                 ):
        super(SATG, self).__init__()
        """
        mode: 0 _ VAD, 1 _ 画像, 2 _ 言語, 3 _ VAD+画像, 4 _ VAD+言語, 5 _ 画像+言語, 6 _ VAD+画像+言語
        """
        self.mode = args.mode
        self.lang = args.lang
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        self.max_frame = args.max_frame
        self.r = args.rate
        
        self.et = 0
        
        self.encoder = Encoder(args, device, input_size, input_img_size, input_p_size, hidden_size)

        if self.mode == 0 or self.mode == 2:
            self.fc = nn.Linear(hidden_size*2, 2)
        elif self.mode == 1:
            self.fc = nn.Linear(hidden_size, 2)
        elif self.mode == 3 or self.mode == 5:
            self.fc = nn.Linear(hidden_size*3, 2)
        elif self.mode == 4:
            self.fc = nn.Linear(hidden_size*4, 2)
        elif self.mode == 7:
            self.fc = nn.Linear(1, 2)
        else:
#             self.fc = nn.Linear(hidden_size*5, 2)
            self.master_transformer = TransformerModel(in_size=hidden_size*5, out_size=2)
            self.ete = ElapsedTimeEncoding(hidden_size*5)
        
    def calc_y(self, h):
#         y = self.fc(h)
        y = self.master_transformer(h)
        return y
        
    def hang_over(self, u, phase='val'):
        """
        u の末端 300 ms を １ にする
        """
        u_ = u.copy()
        if phase=='val':
            for i in range(len(u)-1):
                if u[i] == 0 and u[i+1] == 1:
                    u_[i+1:i+7] = 0
        return u_

    def forward(self, batch, lbl_pre=0, silence=0, phase='train'):
        
        h = self.encoder(batch)
        
        label = torch.tensor(batch['y']).to(self.device, dtype=torch.long)
        u = self.hang_over(batch['u'], phase)
        u = torch.tensor(u).to(self.device, dtype=torch.float32)
        up = batch['u_pred']
#         up = torch.tensor(batch['u_pred']).to(self.device, dtype=torch.float32)

        # 無音の長さ特徴量の作成
        silence_list = []        
        
        for uu in u:
            if uu >= 0.5:
                silence += 1
                silence_list.append(silence)
            else:
                silence = 0
                silence_list.append(silence)
        
        silence_list2 = []
        scnt = 0
        et = 0 #self.et
        u_pre = 0
        for uu in up:
            et += uu
            if uu >= 0.5:
                scnt = 0

            if u_pre>=0.5 and uu < 0.5:
                scnt += 1

            elif uu < 0.5 and scnt > 0:
                scnt += 1

            if scnt > 5:    
                et = 0
                scnt = 0

            silence_list2.append(et)
            u_pre = uu
            
        self.et = et
        
        h = self.ete(h, silence_list2)

        y = self.calc_y(h)
        y = y.squeeze(0)

        loss = 0
        flg = True
        cnt = 0

        y_list1, lbl_list1 = [], []
        for i in range(len(y)):
            y_ = y[i]
            u_ = u[i]
            lbl = label[i]
            sl = silence_list[i]
            
            if sl == 1:
                start_silence = i
            elif sl == 0:
                flg=True
                
            if lbl_pre==0 and lbl==1:
                if sl<3 or sl>40:
                    flg=False
                    
            if (u_>=0.5 and flg) or u_ < 0.5:
                y_list1.append(y_.unsqueeze(0))
                lbl_list1.append(lbl.unsqueeze(0))
                
            lbl_pre = lbl
        
        if len(y_list1)>0:
            cnt+=1
            y_batch = torch.cat(y_list1)
            lbl_batch = torch.cat(lbl_list1)
            loss += self.criterion(y_batch, lbl_batch)
        
        y = self.softmax(y).cpu().data.numpy()[:,1]

        return {'y': y, 'silence': silence_list, 'loss': loss, 'lbl_pre': lbl, 'cnt': cnt}
    
#     def get_attention_weight(self, batch, transformer_type="master"):
#         attention_weight = self.encoder.get_attention_weight(batch)
#         return attention_weight

    def get_attention_weight(self, batch, silence=0, transformer_type="master"):
        label = torch.tensor(batch['y']).to(self.device, dtype=torch.long)
        u = self.hang_over(batch['u'], phase='val')
        u = torch.tensor(u).to(self.device, dtype=torch.float32)
        up = torch.tensor(batch['u_pred']).to(self.device, dtype=torch.float32)

        # 無音の長さ特徴量の作成
        silence_list = []        
        
        for uu in u:
            if uu >= 0.5:
                silence += 1
                silence_list.append([silence])
            else:
                silence = 0
                silence_list.append([silence])
        
        silence_list = torch.FloatTensor(silence_list).to(self.device)
        silence_list = silence_list.unsqueeze(0)
        
        attention_weight = self.encoder.get_attention_weight(batch)
#         attention_weight = self.encoder.get_attention_weight(batch, silence_list)
        silence_list = silence_list.reshape(-1).cpu().data.numpy()        
        
        return attention_weight, silence_list
    
    def get_master_attention_weight(self, batch, silence=0, transformer_type="master"):
        
        label = torch.tensor(batch['y']).to(self.device, dtype=torch.long)
        u = self.hang_over(batch['u'], phase='val')
        u = torch.tensor(u).to(self.device, dtype=torch.float32)
        up = torch.tensor(batch['u_pred']).to(self.device, dtype=torch.float32)

        # 無音の長さ特徴量の作成
        silence_list = []  
        
        for uu in u:
            if uu >= 0.5:
                silence += 1
                silence_list.append([silence])
            else:
                silence = 0
                silence_list.append([silence])
        
        silence_list = torch.FloatTensor(silence_list).to(self.device)
        silence_list = silence_list.unsqueeze(0)
        
        h = self.encoder(batch)
#         h = self.encoder(batch, silence_list)
        silence_list = silence_list.reshape(-1).cpu().data.numpy()
        h = self.silence_encoding(h, silence_list)

        attention_weight = self.master_transformer.get_attention_weight(h)
        
        return attention_weight

    def reset_state(self):
        self.et = 0
        self.encoder.reset_state()
        
    def back_trancut(self):
        self.encoder.back_trancut()

    def reset_lang(self):
        self.encoder.reset_lang()


class Encoder(nn.Module):
    def __init__(self,
                 args,
                 device,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=128,
                 hidden_size=64,
                 ):
        super(Encoder, self).__init__()
        
        self.mode = args.mode
        self.lang = args.lang
        self.device = device
        
        self.input_size = input_size
        self.input_img_size = input_img_size
        self.input_p_size = input_p_size
        self.hidden_size = hidden_size

        # 音響Transformer
        if self.mode in [0, 3, 4, 6]:
            self.transformer_vA = TransformerModel(in_size=input_size, out_size=hidden_size)
            self.transformer_vB = TransformerModel(in_size=input_size, out_size=hidden_size)

        # 画像Transformer
        if self.mode in [1, 3, 5, 6]:
            self.transformer_i = TransformerModel(in_size=input_img_size, out_size=hidden_size)

        # 言語Transformer
        if self.mode in [2, 4, 5, 6]:
            self.transformer_lA = TransformerModel(in_size=input_p_size, out_size=hidden_size)
            self.transformer_lB = TransformerModel(in_size=input_p_size, out_size=hidden_size)
            
        if self.mode in [2, 4, 5, 6]:
            self.la_mem = np.zeros([1, 64])
            self.lb_mem = np.zeros([1, 64])
            self.prev_hla = torch.zeros(1, 1, 64).to(self.device)
            self.prev_hlb = torch.zeros(1, 1, 64).to(self.device)
            
    def calc_voice(self, xA, xB):
        enc_vA = self.transformer_vA(xA)
        enc_vB = self.transformer_vB(xB)
        return enc_vA, enc_vB
    
    def get_voice_attn(self, xA, xB):
        attn_vA = self.transformer_vA.get_attention_weight(xA)
        attn_vB = self.transformer_vB.get_attention_weight(xB)
        return attn_vA, attn_vB

    def calc_img(self, img):
        enc_i = self.transformer_i(img)
        return enc_i
    
    def get_img_attn(self, img):
        attn_i = self.transformer_i.get_attention_weight(img)
        return attn_i

    def calc_lang(self, LA, LB):
        lA_list = []
        lB_list = []
        for i in range(min(len(LA), len(LB))):
            la = LA[i]
            lb = LB[i]
            if len(la) == 0:
                la_out = self.prev_hla
            else:
                if len(la.shape) == 1:
                    la = la.reshape(1, -1)
                    
                self.la_mem = np.concatenate([self.la_mem, la])

                la_in = torch.FloatTensor(self.la_mem).to(self.device)
                la_in = la_in.unsqueeze(0)
                la_out = self.transformer_lA(la_in, N=la_in.size(1))
                self.prev_hla = la_out
                
            lA_list.append(la_out[:, -1, :])

            if len(lb) == 0:
                lb_out = self.prev_hlb
            else:
                if len(lb.shape) == 1:
                    lb = lb.reshape(1, -1)
                    
                self.lb_mem = np.concatenate([self.lb_mem, lb])
            
                lb_in = torch.FloatTensor(self.lb_mem).to(self.device)
                lb_in = lb_in.unsqueeze(0)
                lb_out = self.transformer_lB(lb_in, N=lb_in.size(1))
                self.prev_hlb = lb_out
                
            lB_list.append(lb_out[:, -1, :])

        enc_la = torch.cat(lA_list)
        enc_lb = torch.cat(lB_list)
        self.reset_lang()
        return enc_la.unsqueeze(0), enc_lb.unsqueeze(0)
    
    def get_lang_attn(self, LA, LB):
        phone_listA = [len(self.la_mem)]
        phone_listB = [len(self.lb_mem)]
        for i in range(min(len(LA), len(LB))):
            la = LA[i]
            lb = LB[i]
            if len(la) == 0:
                phone_listA.append(0)
                pass
            else:
                if len(la.shape) == 1:
                    la = la.reshape(1, -1)
                    
                phone_listA.append(len(la))

                self.la_mem = np.concatenate([self.la_mem, la])
                
            if len(lb) == 0:
                phone_listB.append(0)
                pass
            else:
                if len(lb.shape) == 1:
                    lb = lb.reshape(1, -1)

                phone_listB.append(len(lb))
                self.lb_mem = np.concatenate([self.lb_mem, lb])
                
                
        la_in = torch.FloatTensor(self.la_mem).to(self.device)
        lb_in = torch.FloatTensor(self.lb_mem).to(self.device)
        la_in = la_in.unsqueeze(0)
        lb_in = lb_in.unsqueeze(0)
        
        attn_la = self.transformer_lA.get_attention_weight(la_in, N=la_in.size(1))
        attn_lb = self.transformer_lB.get_attention_weight(lb_in, N=lb_in.size(1))

        self.reset_lang()
        return attn_la, attn_lb, (phone_listA, phone_listB)
    
    def forward(self, batch):
        # 1つの入力の時系列の長さ
        self.seq_size = batch['voiceA'].shape[0]
        
        xA = torch.tensor(batch['voiceA']).to(self.device, dtype=torch.float32)
        xA = xA.unsqueeze(0)
        xB = torch.tensor(batch['voiceB']).to(self.device, dtype=torch.float32)
        xB = xB.unsqueeze(0)
        img = torch.tensor(batch['img']).to(self.device, dtype=torch.float32)
        img = img.unsqueeze(0)
        if self.lang == 'ctc' or self.lang == 'julius':
            PA = batch['phonemeA']
            PB = batch['phonemeB']
        else:
            PA = batch['wordA']
            PB = batch['wordB']

        if self.mode in [0, 3, 4, 6]:
            hA, hB = self.calc_voice(xA, xB)

        if self.mode in [1, 3, 5, 6]:
            hImg = self.calc_img(img)

        if self.mode in [2, 4, 5, 6]:
            hPA, hPB = self.calc_lang(PA, PB)
        
        # 特徴量のconcat
        if self.mode == 0:
            h = torch.cat([hA, hB], dim=-1)
        elif self.mode == 1:
            h = hImg
        elif self.mode == 2:
            h = torch.cat([hPA, hPB], dim=-1)
        elif self.mode == 3:
            h = torch.cat([hA, hB, hImg], dim=-1)
        elif self.mode == 4:
            h = torch.cat([hA, hB, hPA, hPB], dim=-1)
        elif self.mode == 5:
            h = torch.cat([hImg, hPA, hPB], dim=-1)
        else:
            h = torch.cat([hA, hB, hImg, hPA, hPB], dim=-1)
        
        return h
    
    def get_attention_weight(self, batch, transformer_type="master"):
        
        self.seq_size = batch['voiceA'].shape[0]
        
        xA = torch.tensor(batch['voiceA']).to(self.device, dtype=torch.float32)
        xA = xA.unsqueeze(0)
        xB = torch.tensor(batch['voiceB']).to(self.device, dtype=torch.float32)
        xB = xB.unsqueeze(0)
        img = torch.tensor(batch['img']).to(self.device, dtype=torch.float32)
        img = img.unsqueeze(0)
        
        attention_weight = {}
        
        if self.lang == 'ctc' or self.lang == 'julius':
            PA = batch['phonemeA']
            PB = batch['phonemeB']
        else:
            PA = batch['wordA']
            PB = batch['wordB']

        if self.mode in [0, 3, 4, 6]:
            attn_vA, attn_vB = self.get_voice_attn(xA, xB)
            attention_weight['voiceA'] = attn_vA
            attention_weight['voiceB'] = attn_vB

        if self.mode in [1, 3, 5, 6]:
            attn_i = self.get_img_attn(img)
            attention_weight['img'] = attn_i

        if self.mode in [2, 4, 5, 6]:
            attn_lA, attn_lB, (phoneA, phoneB) = self.get_lang_attn(PA, PB)
            attention_weight['langA'] = attn_lA
            attention_weight['langB'] = attn_lB
            
            attention_weight['phoneA'] = phoneA
            attention_weight['phoneB'] = phoneB
        
        return attention_weight
    
    def reset_state(self):
        self.la_mem = np.zeros([1, 64])
        self.lb_mem = np.zeros([1, 64])
        
        if self.mode in [0, 3, 4, 6]:
            self.transformer_vA.mem = None
            self.transformer_vA.mem_ = None
            self.transformer_vB.mem = None
            self.transformer_vB.mem_ = None

        # 画像Transformer
        if self.mode in [1, 3, 5, 6]:
            self.transformer_i.mem = None
            self.transformer_i.mem_ = None

        # 言語Transformer
        if self.mode in [2, 4, 5, 6]:
            self.transformer_lA.mem = None
            self.transformer_lA.mem_ = None
            self.transformer_lB.mem = None
            self.transformer_lB.mem_ = None
            
    def back_trancut(self):

        if self.mode in [2, 4, 5, 6]:
            self.prev_hla = self.prev_hla.detach()
            self.prev_hlb = self.prev_hlb.detach()

    def reset_lang(self):
        self.la_mem = np.zeros([1, 64])
        self.lb_mem = np.zeros([1, 64])
        self.prev_hla = torch.zeros(1, 1, 64).to(self.device)
        self.prev_hlb = torch.zeros(1, 1, 64).to(self.device)
