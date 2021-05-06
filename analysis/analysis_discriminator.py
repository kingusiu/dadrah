import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import numpy as np
#import ROOT as rt
#import root_numpy as rtnp
import os
from prettytable import PrettyTable

import anpofah.util.plotting_util as pu
import mplhep as hep

#import dadrah.analysis.root_plotting_util as ropl


def analyze_multi_quantile_discriminator_cut(discriminator_list, sample, feature_key='mJJ', title_suffix='', plot_name='multi_discr_cut', fig_dir=None, cut_xmax=True):

    # Load CMS style sheet
    plt.style.use(hep.style.CMS)
    jet= plt.get_cmap('gnuplot')
    colors = iter(jet(np.linspace(0,1,6)))

    fig = plt.figure()
    x_min = np.min(sample[feature_key])
    if cut_xmax:
        x_max = np.percentile(sample[feature_key], 1e2*(1-1e-4))
    else:
        x_max = np.max(sample[feature_key])
    loss = discriminator_list[0].loss_strategy(sample)
    plt.hist2d(sample[feature_key], loss, range=((x_min , x_max), (np.min(loss), np.percentile(loss, 1e2*(1-1e-3)))), \
                norm=LogNorm(), bins=200, cmap=cm.get_cmap('Blues'), cmin=0.01)
    xs = np.arange(x_min, x_max, 0.001*(x_max-x_min))
    for discriminator in discriminator_list:
        plt.plot(xs, discriminator.predict( xs ) , '-', lw=2.5, label='Q '+str(discriminator.quantile*100)+'%', color=next(colors))
    plt.ylabel('min(L1,L2)')
    plt.xlabel('$M_{jj}$ [GeV]')
    #plt.title('quantile cuts' + title_suffix)
    plt.colorbar()
    plt.legend(loc='best', title='quantile cuts')
    plt.draw()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def analyze_discriminator_cut(discriminator, sample, feature_key='mJJ', plot_name='discr_cut', fig_dir=None):
    fig = plt.figure(figsize=(8, 8))
    x_min = np.min(sample[feature_key])*0.8
    x_max = np.percentile(sample[feature_key], 99.99)
    loss = discriminator.loss_strategy(sample)
    plt.hist2d(sample[feature_key], loss,
           range=((x_min , x_max), (np.min(loss), np.percentile(loss, 1e2*(1-1e-4)))), 
           norm=LogNorm(), bins=100, label='signal data')

    xs = np.arange(x_min, x_max, 0.001*(x_max-x_min))
    plt.plot(xs, discriminator.predict(xs) , '-', color='m', lw=2.5, label='selection cut')
    plt.ylabel('L1 & L2 > LT')
    plt.xlabel('$M_{jj}$ [GeV]')
    plt.title(str(sample) + ' ' + str(discriminator) )
    plt.colorbar()
    plt.legend(loc='best')
    plt.draw()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.png'), bbox_inches='tight')
    plt.close(fig)


def print_discriminator_efficiency_table(sample_dict):
    table = PrettyTable()
    table.field_names = ['Sample', 'Eff VAE [%]']
    for name, sample in sample_dict.items():
        eff = len(sample.accepted()) / float(len(sample))
        table.add_row([name, "{:.2f}".format(eff*100)])
    print(table)


def plot_mass_spectrum_ratio(mjj_bg_like, mjj_sig_like, binning, SM_eff, title='', fig_dir=None, plot_name=None):
    
    h_a = ropl.create_TH1D(mjj_bg_like, name='h_acc', title='BG like', binning=binning, opt='overflow' )
    bin_edges = [h_a.GetXaxis().GetBinLowEdge(i) for i in range(1,h_a.GetNbinsX()+2)]
    h_a.SetLineColor(2)
    h_a.SetStats(0)
    h_a.Sumw2()
    
    h_r = ropl.create_TH1D(mjj_sig_like, name='h_rej', title='SIG like', axis_title=['M_{jj} [GeV]', 'Events'], binning=binning, opt='overflow' )
    bin_edges = [h_a.GetXaxis().GetBinLowEdge(i) for i in range(1,h_a.GetNbinsX()+2)]

    h_r.GetYaxis().SetRangeUser(0.5, 1.2*h_r.GetMaximum())
    h_r.SetStats(0)
    h_r.Sumw2()

    c = ropl.make_effiency_plot([h_r, h_a], ratio_bounds=[1e-4, 0.2], draw_opt = 'E', title=title)

    c.pad1.SetLogy()
    c.pad2.SetLogy()

    c.pad2.cd()
    c.ln = rt.TLine(h_r.GetXaxis().GetXmin(), SM_eff, h_r.GetXaxis().GetXmax(), SM_eff)
    c.ln.SetLineWidth(2)
    c.ln.SetLineStyle(7)
    c.ln.SetLineColor(8)
    c.ln.DrawLine(h_r.GetXaxis().GetXmin(), SM_eff, h_r.GetXaxis().GetXmax(), SM_eff)

    c.Draw()

    if fig_dir is not None:
        c.SaveAs(os.path.join(fig_dir,plot_name))

    #return c
