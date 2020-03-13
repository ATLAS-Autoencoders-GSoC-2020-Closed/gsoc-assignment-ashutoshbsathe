import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle 
from model import AutoEncoder, AE_3D_200
import utils 
from scipy import stats

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
TRAINDATA = 'all_jets_train_4D_100_percent.pkl'
TESTDATA = 'all_jets_test_4D_100_percent.pkl'
N_TRAIN = 111778
N_TEST = 27945
ENCODER_PATH = r'.\runs\20200313_150302_Adam_frac1.0_0.0093\encoder_epoch_799.pt'
DECODER_PATH = r'.\runs\20200313_150302_Adam_frac1.0_0.0093\decoder_epoch_799.pt'

plot_recon = True
plot_hist = True
plot_residuals = True
# stay thread safe
def main():
    test_loss_data = pd.read_csv(r'./tensorboard_data/run-tag-test_loss.csv')
    test_loss_data['Step'] += 1

    # plot the reconstruction loss
    if plot_recon:
        plt.plot(test_loss_data['Step'], test_loss_data['Value'], marker='.', label='Test Set')
        plt.title('Reconstruction loss on test set')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('./plots/recon_loss_test.png', dpi=300)
        plt.clf()
        print('Loss plot generated')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(TRAINDATA, 'rb') as f:
        traindata = pickle.load(f).to_numpy()
    with open(TESTDATA, 'rb') as f:
        testdata = pickle.load(f).to_numpy()
    
    train_min = traindata.min(axis=0)
    train_max = traindata.max(axis=0)

    scaled_train = (traindata - train_min) / (train_max - train_min)
    scaled_test = (testdata - train_min) / (train_max - train_min)

    scaled_train = torch.FloatTensor(scaled_train).to(device)
    scaled_test = torch.FloatTensor(scaled_test).to(device)


    model = AutoEncoder(connections=128).to(device)
    model.encoder.load_state_dict(torch.load(ENCODER_PATH))
    model.encoder.eval()
    model.decoder.load_state_dict(torch.load(DECODER_PATH))
    model.decoder.eval()
    model.eval()
    


    recon_scaled_train = model(scaled_train)
    print('Train set loss: {}'.format(nn.MSELoss()(recon_scaled_train, scaled_train)))
    recon_scaled_train = recon_scaled_train.detach().cpu().numpy()
    recon_scaled_test = model(scaled_test)
    print('Test set loss: {}'.format(nn.MSELoss()(recon_scaled_test, scaled_test)))
    recon_scaled_test = recon_scaled_test.detach().cpu().numpy()

    recon_train = recon_scaled_train * (train_max - train_min) + train_min
    recon_test = recon_scaled_test * (train_max - train_min) + train_min 

    colors = ['orange', 'c']
    unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
    columns = []
    variable_list = [r'$m$', r'$p_T$', r'$\phi$', r'$\eta$']
    line_style = ['--', '-']
    markers = ['*', 's']
    if plot_hist:
        alph = 0.75
        n_bins = 10
        data = testdata
        pred = recon_test
        for kk in np.arange(4):
            plt.clf()
            plt.figure(kk + 4)
            n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
            n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
            plt.suptitle('Histogram of {} on test data'.format(variable_list[kk]))
            plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
            plt.ylabel('Number of events')
            plt.legend()
            plt.savefig('./plots/testdata_{}.png'.format(kk))
        data = traindata
        pred = recon_train
        for kk in np.arange(4):
            plt.clf()
            plt.figure(kk + 4)
            n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
            n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
            plt.suptitle('Histogram of {} on train data'.format(variable_list[kk]))
            plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
            plt.ylabel('Number of events')
            plt.legend()
            plt.savefig('./plots/traindata_{}.png'.format(kk))
        print('Histograms plot completed')
    """
    TODO: clean up the code for residuals
    if plot_residuals:
        pred = recon_train 
        data = traindata 
        alph = 0.75
        residual_strings = [r'$(p_{T,out} - p_{T,in}) / p_{T,in}$',
                        r'$(\eta_{out} - \eta_{in}) / \eta_{in}$',
                        r'$(\phi_{out} - \phi_{in}) / \phi_{in}$',
                        r'$(E_{out} - E_{in}) / E_{in}$']
        residuals = (pred - data) / data
        range = (-.02, .02)
        #range=None
        for kk in np.arange(4):
            plt.figure()
            n_hist_pred, bin_edges, _ = plt.hist(
                residuals[:, kk], label='Residuals', linestyle=line_style[0], alpha=alph, bins=10, range=range)
            plt.suptitle('Residuals of {}'.format(variable_list[kk]))
            plt.xlabel(residual_strings[kk])  # (train.columns[kk], train.columns[kk], train.columns[kk]))
            plt.ylabel('Number of jets')
            #plt.yscale('log')
            std = np.std(residuals[:, kk])
            std_err = utils.std_error(residuals[:, kk])
            mean = np.nanmean(residuals[:, kk])
            sem = stats.sem(residuals[:, kk], nan_policy='omit')
            ax = plt.gca()
            plt.text(.75, .8, 'Mean = %f$\pm$%f\n$\sigma$ = %f$\pm$%f' % (mean, sem, std, std_err), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
            plt.savefig('./plots/residuals_traindata_{}.png'.format(kk))
    """

if __name__ == '__main__':
    main()