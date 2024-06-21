import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date

from model.model import MTHCCAR
from utils.data_loader import Preprocessing

dir_data = './data/rt_nn_cloud_training_data_20231016.nc'
dir_models = './models'
dir_results = './results'

Path(dir_models).mkdir(parents=True, exist_ok=True)
Path(dir_results).mkdir(parents=True, exist_ok=True)

def train_model(
        model,
        epochs: int = 500,
        batch_size: int = 64,
        learning_rate: float = 1e-5,
        cls_w: float = 0.6,
        exp: str = f'MT_HCCAR_{date.today().month}{date.today().day}'
):
    # 1. Create dataset
    X_train, y_train, X_val, y_val = Preprocessing(dir_data)

    # 2. Create data loaders
    dataset_train = TensorDataset(X_train, y_train)
    dataset_val = TensorDataset(X_val, y_val)
    train_loader = DataLoader(dataset = dataset_train, batch_size = batch_size, shuffle=True, pin_memory=True)
    validate_loader = DataLoader(dataset = dataset_val, batch_size = batch_size, shuffle=True, pin_memory=True)

    # 3. Set up the optimizer, the loss, the learning rate 
    loss_recon = nn.L1Loss()
    loss_cls = nn.BCELoss()
    loss_clsaux = nn.CrossEntropyLoss()
    loss_reg = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Begin training
    train_loss_all = []
    valid_loss_all = []
    train_loss_cls1 = []
    train_loss_cls2 = []
    train_loss_cls3 = []
    train_loss_reg = []
    valid_loss_cls1 = []
    valid_loss_cls2 = []
    valid_loss_cls3 = []
    valid_loss_reg = []
    ## Max and minvalue of COT, to recover COT to its original range
    max_value = Preprocessing.max_value
    min_value = Preprocessing.min_value

    ## Select model based on validation loss
    valid_loss_min = 10.0

    file = open(f'{dir_results}/{exp}oci_log.txt',"w")

    for epoch in range(epochs): 
    
        # Set current loss value
        train_loss = 0.0
        train_cls1 = 0.0
        train_cls2 = 0.0
        train_cls3 = 0.0
        train_reg = 0.0

        valid_loss = 0.0
        valid_cls1 = 0.0
        valid_cls2 = 0.0
        valid_cls3 = 0.0
        valid_reg = 0.0

        train_num = 0
        valid_num = 0
        
        # Iterate over the DataLoader for training data
        for step, (b_x,b_y) in enumerate(train_loader):
            # Get predictions
            recon, output_cls1, output_cls2, output_cls3, cloud_mask, output_reg = model(b_x)
            output_reg = output_reg*(max_value-min_value)+min_value
            cloud_n = torch.sum(cloud_mask)+1
            minibatch_n = output_cls1.shape[0]

            # Get ground truth values
            ground_truth_cls1 = torch.reshape(b_y[:,0], [-1,1]) # Cloud-free = 1, Cloudy = 0
            ground_truth_cls2 = torch.reshape(b_y[:,1], [-1,1]) # Liquid = 1, Ice = 0
            ground_truth_reg = torch.reshape(b_y[:,3], [-1,1])
            ground_truth_reg = ground_truth_reg*(max_value-min_value)+min_value
            cls3_1 = torch.where((ground_truth_reg<0)&(ground_truth_reg>-2), 1., 0.)
            cls3_2 = torch.where((ground_truth_reg>=0)&(ground_truth_reg<1), 1., 0.)
            cls3_3 = torch.where((ground_truth_reg>=1)&(ground_truth_reg<=2.5), 1., 0.)
            ground_truth_cls3 = torch.concatenate((cls3_1, cls3_2, cls3_3), 1)
            ground_truth_cls3 = ground_truth_cls3.float()

            # Loss functions
            l_recon = loss_recon(recon, b_x)
            l_cls1 = loss_cls(output_cls1, ground_truth_cls1)
            l_cls2 = loss_cls(output_cls2, ground_truth_cls2)*minibatch_n/cloud_n
            l_cls3 = loss_clsaux(output_cls3, ground_truth_cls3)*minibatch_n/cloud_n
            l_reg = loss_reg(output_reg, ground_truth_reg)*minibatch_n/cloud_n

            ## Compute L1 loss component
            l1_weight = 0.0002
            l1_parameters = []
            for parameter in model.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = l1_weight*torch.abs(torch.cat((l1_parameters))).sum()

            loss = 3*l_recon + cls_w * (l_cls1 + l_cls2 + l_cls3) + (1-cls_w)*l_reg + 0.5*l1

            l_cls1.requires_grad_(True)
            l_cls2.requires_grad_(True)
            l_cls3.requires_grad_(True)
            l_reg.requires_grad_(True)
            file.write(f'Train recon Loss: {l_recon:.3f}, Train CLS Loss 1: {l_cls1:.3f}, Train CLS Loss 2: {l_cls2:.3f}, Train CLS Loss 3: {l_cls3:.3f}, Train REG Loss: {l_reg:.3f}\n') 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss +=loss.item() * b_x.size(0)
            train_num += b_x.size(0)
            train_cls1 += l_cls1.item() * b_x.size(0)
            train_cls2 += l_cls2.item() * b_x.size(0)
            train_cls3 += l_cls3.item() * b_x.size(0)
            train_reg += l_reg.item() * b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        train_loss_cls1.append(train_cls1 / train_num)
        train_loss_cls2.append(train_cls2 / train_num)
        train_loss_cls3.append(train_cls3 / train_num)
        train_loss_reg.append(train_reg / train_num)

        for setp, (c_x, c_y) in enumerate(validate_loader):
            # Get predictions
            recon_valid, output_cls1_valid, output_cls2_valid, output_cls3_valid, cloud_mask_valid, output_reg_valid = model(c_x)
            output_reg_valid = output_reg_valid*(max_value-min_value)+min_value
            cloud_n = torch.sum(cloud_mask_valid)+1
            minibatch_n = output_cls1_valid.shape[0]

            # Get ground truth values
            ground_truth_cls1_valid = torch.reshape(c_y[:,0], [-1,1])
            ground_truth_cls2_valid = torch.reshape(c_y[:,1], [-1,1])
            ground_truth_reg_valid = torch.reshape(c_y[:,3], [-1,1])
            ground_truth_reg_valid = ground_truth_reg_valid*(max_value-min_value)+min_value
            cls3_1_v = torch.where((ground_truth_reg_valid<0), 1., 0.)
            cls3_2_v = torch.where((ground_truth_reg_valid>=0)&(ground_truth_reg_valid<1), 1., 0.)
            cls3_3_v = torch.where((ground_truth_reg_valid>=1), 1., 0.)
            ground_truth_cls3_v = torch.concatenate((cls3_1_v, cls3_2_v, cls3_3_v), 1)

            l_recon_v = loss_recon(recon_valid, c_x)
            l_cls1_v = loss_cls(output_cls1_valid, ground_truth_cls1_valid)
            l_cls2_v = loss_cls(output_cls2_valid, ground_truth_cls2_valid)*minibatch_n/cloud_n
            l_cls3_v = loss_clsaux(output_cls3_valid, ground_truth_cls3_v)*minibatch_n/cloud_n
            l_reg_v = loss_reg(output_reg_valid, ground_truth_reg_valid)*minibatch_n/cloud_n
            file.write(f'Valid CLS Loss 1: {l_cls1_v:.3f}, Valid CLS Loss 2: {l_cls2_v:.3f}, Valid REG Loss: {l_reg_v:.3f}\n') 
            
            loss_v = 3*l_recon_v + cls_w * (l_cls1_v + l_cls2_v + l_cls3_v) + (1-cls_w)*l_reg_v + 0.5*l1

            valid_loss +=loss_v.item() * c_x.size(0)
            valid_num += c_x.size(0)
            valid_cls1 += l_cls1_v.item() * c_x.size(0)
            valid_cls2 += l_cls2_v.item() * c_x.size(0)
            valid_cls3 += l_cls3_v.item() * c_x.size(0)
            valid_reg += l_reg_v.item() * c_x.size(0)
        valid_loss_all.append(valid_loss / valid_num)
        valid_loss_cls1.append(valid_cls1 / valid_num)
        valid_loss_cls2.append(valid_cls2 / valid_num)
        valid_loss_cls3.append(valid_cls3 / valid_num)
        valid_loss_reg.append(valid_reg / valid_num)

        print(f'Epoch {epoch+1} - Train Loss: {(train_loss / train_num):.3f}, Valid Loss: {(valid_loss / valid_num):.3f}')  
        file.write(f'Epoch {epoch+1} - Train Loss: {(train_loss / train_num):.3f}, Valid Loss: {(valid_loss / valid_num):.3f}, l1 Loss: {l1:.3f} \n')

        if (valid_loss / valid_num) <= valid_loss_min:
            valid_loss_min = (valid_loss / valid_num)
            torch.save(model.state_dict(), f'{dir_models}/{exp}_min_valid_loss.pth')
    
    file.write(f'Minimum Valid Loss: Epoch {epoch} - {valid_loss_min:.3f} \n')
    file.close()
    print('Training process has finished.')

    # 5. Save loss function plots
    plt.figure(figsize = (8,6))
    plt.plot(train_loss_all, 'r-', label = 'Training loss')
    plt.plot(valid_loss_all, 'b-', label = 'Validation loss')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{dir_results}/{exp}_loss.png')
    # plt.show()

    plt.figure(figsize = (8,6))
    plt.plot(train_loss_cls1, 'r-', label = 'Training classification loss 1')
    plt.plot(train_loss_reg, 'b-', label = 'Training regression loss')
    plt.plot(train_loss_cls2, 'g-', label = 'Training classification loss 2')
    plt.plot(train_loss_cls3, 'y-', label = 'Training classification loss 3')
    plt.plot(valid_loss_cls1, 'r--', label = 'Validation classification loss 1')
    plt.plot(valid_loss_cls2, 'g--', label = 'Validation classification loss 2')
    plt.plot(valid_loss_cls3, 'y--', label = 'Validation classification loss 3')
    plt.plot(valid_loss_reg, 'b--', label = 'Validation regression loss')

    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Loss (classification and regression)')
    plt.savefig(f'{dir_results}/{exp}_loss_detail.png')
    # plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='Train the MT-HCCAR on OCI dataset')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--loss-weight-of-cls', '-w', metavar='WCLS', type=float, default=0.6,
                        help='The weight of classifications task in loss function', dest='w_cls')
    parser.add_argument('--experiment-name', '-n', metavar='EXP', type=str, default='mthccar_oci',
                        help='The name of experiment', dest='exp')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTHCCAR(input_size=233)
    # if args.load:
    #     state_dict = torch.load(args.load, map_location=device)
    #     del state_dict['mask_values']
    #     model.load_state_dict(state_dict)
    model.to(device=device)
    train_model(
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            cls_w=args.w_cls,
            exp=args.exp
        )