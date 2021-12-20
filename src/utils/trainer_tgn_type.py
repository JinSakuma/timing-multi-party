import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from src.utils.eval import evaluation


def train(net, mode, dataloaders_dict,
          device, optimizer, scheduler
          ):

    dataloaders_dict['train'].on_epoch_end()
    net.train()
    
    if scheduler is not None: 
        scheduler.step()
        print('lr:{}'.format(scheduler.get_last_lr()[0]))

    epoch_loss = 0.0
    train_cnt = 0

    for batch in tqdm(dataloaders_dict['train']):
        y_pre, a_pre = 0, 0
        net.reset_state()
        if mode == 2 or mode >= 4:
            net.reset_lang()

        for inputs in batch:
            output_dict = net(inputs, y_pre, a_pre)
            y_pre = output_dict['y'][-1]
            a_pre = output_dict['alpha'][-1]

            loss = output_dict['loss']
            if loss != 0 and loss != -1:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                net.back_trancut()
                loss = loss.item()

            epoch_loss += loss
            loss = 0
            train_cnt += output_dict['cnt']

    epoch_loss = epoch_loss / train_cnt
    output_dict = {}

    return epoch_loss


def val(net, mode, dataloaders_dict,
        device, optimizer, scheduler,
        epoch, output='./', resume=False
        ):

    #  dataloaders_dict['val'].on_epoch_end()
    net.eval()
    epoch_loss = 0.0
    train_cnt = 0
    threshold = 0.8
    a_pred = []
    u_true, u_pred, u_pred_hat = [], [], []
    y_true, y_pred = [], []
    endpoint = []

    with torch.no_grad():
        for batch in tqdm(dataloaders_dict['val']):
            y_pre, a_pre = 0, 0
            net.reset_state()
            if mode == 2 or mode >= 4:
                net.reset_lang()

            for inputs in batch:
                output_dict = net(inputs, y_pre, a_pre, phase='val')
                y_pre = output_dict['y'][-1]
                a_pre = output_dict['alpha'][-1]

                a_pred.append(output_dict['alpha'])
                u_true.append(inputs['u'])
                u_pred.append(output_dict['u_pred'])
                u_pred_hat.append(output_dict['u_pred_hat'])
                y_true.append(inputs['y'])
                y_pred.append(output_dict['y'])
                endpoint.append(inputs['endpoint'])

                loss = output_dict['loss']
                if loss != 0 and loss != -1:
                    net.back_trancut()
                    loss = loss.item()

                epoch_loss += loss
                loss = 0
                train_cnt += output_dict['cnt']

        epoch_loss = epoch_loss / train_cnt
        if resume:
            fig = plt.figure(figsize=(20, 8))
            plt.rcParams["font.size"] = 18
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)

            N=len(u_true[0])
            ax1.plot([threshold]*N, color='black', linestyle='dashed')
            ax1.plot(u_true[0], label='u_true', color='g', linewidth=3.0)
            ax1.plot(u_pred[0], label='u_pred', color='g', linewidth=2.0)
            ax1.plot(u_pred_hat[0], label='u_pred_hat', color='m', linewidth=2.0)
            ax1.legend()
            # plt.fill_between(range(300), u_pred[:300], color='g', alpha=0.3)
            ax2.plot([threshold]*N, color='black', linestyle='dashed')
            ax2.plot(y_pred[0], label='predict', linewidth=3.0)
            ax2.plot(a_pred[0], label='a_t', color='r', linewidth=3.0)
            ax2.fill_between(range(N), a_pred[0], color='r', alpha=0.3)
            ax2.plot(y_true[0], label='true label', linewidth=4.0, color='b')
            ax2.legend()
            plt.savefig(os.path.join(output, 'result_{}_loss_{:.3f}.png'.format(epoch+1, epoch_loss)))
            plt.close()
            
            label = []
            pred = []
            uttr = []
            for i in range(len(y_true)):
                label.append(y_true[i][endpoint[i]:])
                pred.append(y_pred[i][endpoint[i]:])
                uttr.append(u_true[i][endpoint[i]:])
                
            precision, recall, f1, MAE = evaluation(label, pred, uttr)

            precision, recall, f1, MAE = evaluation(y_true, y_pred, u_true)
            torch.save(net.state_dict(), os.path.join(output, 'epoch_{}_loss_{:.4f}_f0.4{:.3f}.pth'.format(epoch+1, epoch_loss, f1)))
            print('-------------')

    return epoch_loss, precision, recall, f1, MAE


def trainer(net,
            device,
            mode,
            dataloaders_dict,
            optimizer, scheduler,
            num_epochs=10,
            output='./',
            resume=False,
            wandb_flg=True
            ):

    os.makedirs(output, exist_ok=True)
    Loss = {'train': [], 'val': []}
    net.to(device)
    
    if wandb_flg:
        wandb.watch(net)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                train_loss = train(net, mode, dataloaders_dict, device, optimizer, scheduler)
                print('{} Loss: {:.4f}'.format(phase, train_loss))
                Loss[phase].append(train_loss)
            else:
                val_loss, val_p, val_r, val_f, val_mae = val(net, mode, dataloaders_dict, device,
                                                            optimizer, scheduler, epoch, output, resume)
                print('{} Loss: {:.4f}'.format(phase, val_loss))
                Loss[phase].append(val_loss)
            
        if wandb_flg:
            wandb.log({
                "Train Loss": train_loss,
                "Valid Loss": val_loss,
                "Valid MAE": val_mae,
                "Valid Precision": val_p,
                "Valid Recall": val_r,
                "Valid F1": val_f
            })

    if resume:
        plt.figure(figsize=(15, 4))
        plt.rcParams["font.size"] = 15
        plt.plot(Loss['val'], label='val')
        plt.plot(Loss['train'], label='train')
        plt.legend()
        plt.savefig(os.path.join(output, 'history.png'))
