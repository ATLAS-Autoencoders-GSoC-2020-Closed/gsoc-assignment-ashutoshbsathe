import torch 
import torch.nn as nn
from utils import get_data_batches 
from tqdm import tqdm
from logger import Logger
from model import AutoEncoder, AE_3D_200
from datetime import datetime
import itertools 
"""
TODO:
1. Setup an outsider config file
2. Add checks for missing directories and create the missing directories as required
3. Clean up the code
"""
def adjust_learning_rate(optimizer, epoch, orig_lr):
    """Sets the learning rate to the initial LR decayed by 5 every 100 epochs"""
    lr = orig_lr * (0.2 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# main wrapper to stay thread safe
def main():
    torch.manual_seed(1618)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using PyTorch Device : {}'.format(device.upper()))

    n_epochs = 800
    LOGDIR = './runs/' + datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = Logger(log_dir=LOGDIR)
    criterion = nn.MSELoss().to(device)
    lr = 1e-4

    model = AutoEncoder(connections=128).to(device)
    optim = torch.optim.Adam(itertools.chain(model.encoder.parameters(), model.decoder.parameters()), lr=lr)
    #model = AE_3D_200().to(device)
    #optim = torch.optim.Adam(itertools.chain(model.encoder.parameters(), model.decoder.parameters()), lr=lr, weight_decay=1e-6)
    train_batch, val_batch, test_batch = get_data_batches(device=device, frac=1.0)
    print(train_batch.size())
    print(val_batch.size())
    print(test_batch.size())
    worst_case_loss = torch.FloatTensor([float('Inf')]).to(device)
    pbar = tqdm(range(n_epochs), leave=True)
    for e in pbar:
        new_lr = lr * (0.2 ** ((e+1) // 100))
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

        optim.zero_grad()
        recon_batch = model(train_batch)
        loss = criterion(recon_batch, train_batch)
        loss.backward()
        optim.step()

        model.eval()
        recon_val = model(val_batch)
        val_loss = nn.MSELoss()(recon_val, val_batch)

        recon_test = model(test_batch)
        test_loss = nn.MSELoss()(recon_test, test_batch)
        model.train()

        info = {
            'train_loss': loss.item(),
            'val_loss': val_loss.item(),
            'test_loss': test_loss.item()
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, e)
        
        
        torch.save(model.encoder.state_dict(), LOGDIR + '/encoder_epoch_{}.pt'.format(e))
        torch.save(model.decoder.state_dict(), LOGDIR + '/decoder_epoch_{}.pt'.format(e))

        pbar.set_description('train_loss: {:.4f}, val_loss: {:.4f}, test_loss: {:.4f}'.format(loss.item(), val_loss.item(), test_loss.item()))


if __name__ == '__main__':
    main()
