import numpy as np
import datetime
import os
import torch
import torch.optim as optim
import argparse
import wandb

from src.models.RTG.rtg_upred import RTG
from src.utils.utils import set_random_seeds
from src.utils.trainer_rtg_type import trainer
from src.utils.data import get_dataloader
from conf.parser import Config


CONF="rtg_upred.yaml"

def main():
    
    args = Config.get_cnf(CONF)

    os.makedirs(args.output, exist_ok=True)
    set_random_seeds(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    
    name = args.exp_name
    out = os.path.join(args.output, name)
    os.makedirs(out, exist_ok=True)
    
    if args.wandb_flg:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=name)

    # モデル設定
    input_size = args.input_size
    input_img_size = args.input_img_size
    hidden_size = args.hidden_size
    input_p_size = args.input_p_size

    print('data loading ...')
    dataloaders_dict = get_dataloader(args)
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using', device)
    net = RTG(args,
             device=device,
             input_size=input_size,
             input_img_size=input_img_size,
             input_p_size=input_p_size,
             hidden_size=hidden_size
             )


    modal_list = ["音響", "画像", "音素", "音響+画像", "音響+音素", "画像+音素", "音響+画像+音素", "経過時間のみ"]
    print('特徴量: {}'.format(modal_list[args.mode]))
    print('Model :', net.__class__.__name__)

#     optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.001)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25], gamma=0.1)
    scheduler = None

    for name, param in net.named_parameters():
        if 'swt' in name or 'bert' in name:
            param.requires_grad = False
            print("勾配計算なし。学習しない：", name)
        else:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)

    print('train data is ', len(dataloaders_dict['train']))
    print('test data is ', len(dataloaders_dict['val']))

    trainer(
        net=net,
        device=device,
        mode=args.mode,
        dataloaders_dict=dataloaders_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epoch,
        output=out,
        resume=args.resume,
        wandb_flg=args.wandb_flg
        )


if __name__ == '__main__':
    main()
