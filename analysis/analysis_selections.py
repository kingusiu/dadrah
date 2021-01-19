from prettytable import PrettyTable
import ROOT as rt
import root_numpy as rtnp

import dadrah.analysis.root_plotting_util as ropl


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
    
    h_r = ropl.create_TH1D(mjj_sig_like, name='h_rej', title='SIG like', axis_title=['M_{jj} [GeV]', 'Events'], binning=binning,
                      opt='overflow' )
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

