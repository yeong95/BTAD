# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2, resnet101
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset, BatteryDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test, evaluation_battery
from torch.nn import functional as F
import pickle
import wandb




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(_class_):
    run = wandb.init(project='RD4AD')
    print(_class_)
    epochs = 100
    learning_rate = 0.005
    batch_size = 16
    image_size = 224
    network = 'res50'
    wandb.config.update({'epochs':epochs, 'learning_rate':learning_rate, 'batch_size':batch_size, 'image_size':image_size,\
         'class':_class_, 'network':network})
        
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = '../무지부_코팅부_테이프제외2/' + _class_
    test_path = '../무지부_코팅부_테이프제외2/' + _class_
    
    os.makedirs('./checkpoints', exist_ok=True)
    train_data = BatteryDataset(root=train_path, transform=data_transform, phase="train")    
    test_data = BatteryDataset(root=test_path, transform=data_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = resnet50(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet50(pretrained=False)
    decoder = decoder.to(device)
    
    # load checkpoints
    # artifact = run.use_artifact('yeong/RD4AD/model:v39', type='model')
    # artifact_dir = artifact.download()
    # model_name = 'wres50_'+_class_+'epochs100'+'.pth'
    # bn.load_state_dict(torch.load(os.path.join(artifact_dir, model_name))['bn'])
    # decoder.load_state_dict(torch.load(os.path.join(artifact_dir, model_name))['decoder'])
    # print(f"RD4AD model loaded from... {model_name}")
    # bn.load_state_dict(torch.load(f'checkpoints/{model_name}')['bn'])
    # decoder.load_state_dict(torch.load(f'checkpoints/{model_name}')['decoder'])

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    train_loss = []
    prauc_list = []
    auroc_list = []

    best_valid = 0
    for epoch in range(epochs):
        ckp_path = './checkpoints/' + network +_class_+'.pth'
        bn.train()
        decoder.train()
        loss_list = []
        for img, _, _, _ in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))#bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        wandb.log({'train_loss':np.mean(loss_list)}, step=epoch)
        # train_loss.append(np.mean(loss_list))
        if (epoch+1) % 10 == 0:
            auroc_sp, prauc_sp = evaluation_battery(encoder, bn, decoder, test_dataloader, device, _class_, epoch+1)
            print('Sample Auroc{:.3f}, Sample Prauc{:.3}'.format(auroc_sp, prauc_sp))
            wandb.log({'sample auroc': auroc_sp, 'sample prauc':prauc_sp}, step=epoch)
            
            if prauc_sp > best_valid:
                torch.save({'bn': bn.state_dict(),
                            'decoder': decoder.state_dict()}, ckp_path)

                # artifact = wandb.Artifact('model', type='model')
                # artifact.add_file(ckp_path)
                # run.log_artifact(artifact)

                best_valid = prauc_sp
                best_idx = epoch+1
                print('==> best model saved - %d epoch / auroc %.3f'%(best_idx, best_valid))
            # prauc_list.append(prauc_sp)
            # auroc_list.append(auroc_sp)
    
    # os.makedirs(f'{_class_}_result', exist_ok=True)
    # plot = {'train_loss' : train_loss, 'prauc_list' : prauc_list, 'auroc_list': auroc_list}
    # with open(f'{_class_}_result/plot.pickle', 'wb') as f:
    #     pickle.dump(plot, f)
   
    return auroc_sp




if __name__ == '__main__':

    setup_seed(111)
    item_list = ['코팅부']
    for i in item_list:
        train(i)
        
