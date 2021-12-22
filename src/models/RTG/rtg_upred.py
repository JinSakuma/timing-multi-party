import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RTG(nn.Module):
    def __init__(self,
                 args,
                 device,
                 input_size=128,
                 input_img_size=65,
                 input_p_size=128,
                 hidden_size=64,
                 ):
        super(RTG, self).__init__()
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
            self.fc = nn.Linear(hidden_size*5+1, 2)
        
    def calc_y(self, h):
        y = self.fc(h)
        return y
        
    def hang_over(self, u, phase='val'):
        """
        u の末端 300 ms を １ にする
        """
        u_ = u.copy()
#         if phase=='val':
        for i in range(len(u)-1):
            if u[i] == 0 and u[i+1] == 1:
                u_[i+1:i+7] = 0
                
        return u_

    def forward(self, batch, lbl_pre=0, silence=0, phase='train'):
        
        output = self.encoder(batch)
        
        label = torch.tensor(batch['y']).to(self.device, dtype=torch.long)
        u = self.hang_over(batch['u'], phase)
        u = torch.tensor(u).to(self.device, dtype=torch.float32)
        up = torch.tensor(batch['u_pred']).to(self.device, dtype=torch.float32)

        # 無音の長さ特徴量の作成
        silence_list = []        
        
        for uu in up:
            if uu >= 0.5:
                silence += 1
                silence_list.append([silence])
            else:
                silence = 0
                silence_list.append([silence])
        
        silence_list = torch.FloatTensor(silence_list).to(self.device)
        silence_list = silence_list.unsqueeze(0)
        
        h = torch.cat([output, silence_list], dim=-1)
        silence_list = silence_list.reshape(-1).cpu().data.numpy()

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
    
    def get_attention_weight(self, batch, transformer_type="master"):
        attention_weight = self.encoder.get_attention_weight(batch)
        return attention_weight

    def reset_state(self):
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

        # 音響LSTM
        if self.mode in [0, 3, 4, 6]:
            self.lstm_v = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
            )

        # 画像LSTM
        if self.mode in [1, 3, 5, 6]:
            self.lstm_i = torch.nn.LSTM(
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
            
        if self.mode in [2, 4, 5, 6]:
            self.prev_hpa = torch.zeros(1, 64).to(self.device)
            self.prev_hpb = torch.zeros(1, 64).to(self.device)
            self.PAD = -1

        self.hidden_size = hidden_size
        self.hidden = None
        self.hiddenA = None
        self.hiddenB = None
        self.hiddenPA = None
        self.hiddenPB = None
        self.hidden_img = None
            
    def calc_voice(self, xA, xB):
        hA, self.hiddenA = self.lstm_v(xA, self.hiddenA)
        hB, self.hiddenB = self.lstm_v(xB, self.hiddenB)
        return hA, hB

    def calc_img(self, img):
        hImg, self.hidden_img = self.lstm_i(img, self.hidden_img)
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
    
    def reset_state(self):
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