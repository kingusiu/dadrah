import numpy as np
import ROOT as rt
import root_numpy as rtnp


def create_TH1D(x, name='h', title=None, binning=[None, None, None], weights=None, h2clone=None, axis_title = ['',''], opt=''):
    if title is None:
        title = name
    if (x.shape[0] == 0):
        # print 'Empty sample while creating TH1D'
        h = rt.TH1D(name, title, 1, 0, 1)
    elif not h2clone is None:
        h = h2clone.Clone(name)
        h.SetTitle(title)
        h.Reset()
    elif isinstance(binning, np.ndarray):
        h = rt.TH1D(name, title, len(binning)-1, binning)
    elif len(binning) == 3:
        if binning[1] is None:
            binning[1] = min(x)
        if binning[2] is None:
            if ((np.percentile(x, 95)-np.percentile(x, 50))<0.2*(max(x)-np.percentile(x, 95))):
                binning[2] = np.percentile(x, 90)
            else:
                binning[2] = max(x)
        if binning[0] is None:
            bin_w = 4*(np.percentile(x,75) - np.percentile(x,25))/(len(x))**(1./3.)
            if bin_w == 0:
                bin_w = 0.5*np.std(x)
            if bin_w == 0:
                bin_w = 1
            binning[0] = int((binning[2] - binning[1])/bin_w) + 5

        h = rt.TH1D(name, title, binning[0], binning[1], binning[2])
    else:
        print('Binning not recognized')
        raise

    if 'underflow' in opt:
        m = h.GetBinCenter(1)
        x = np.copy(x)
        x[x<m] = m
    if 'overflow' in opt:
        M = h.GetBinCenter(h.GetNbinsX())
        x = np.copy(x)
        x[x>M] = M

    rtnp.fill_hist(h, x, weights=weights)
    h.SetLineWidth(2)
    h.SetXTitle(axis_title[0])
    h.SetYTitle(axis_title[1])
    h.binning = binning
    return h


def make_effiency_plot(h_list_in, title = "", label = "", in_tags = None, ratio_bounds = [None, None], draw_opt = 'P', canvas_size=(600,600)):
    h_list = []
    if in_tags == None:
        tag = []
    else:
        tag = in_tags
    for i, h in enumerate(h_list_in):
        h_list.append(h.Clone('h{}aux{}'.format(i, label)))
        if in_tags == None:
            tag.append(h.GetTitle())

    c_out = rt.TCanvas("c_out_ratio"+label, "c_out_ratio"+label, canvas_size[0], canvas_size[1])
    pad1 = rt.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0.03)
    pad1.SetLeftMargin(0.13)
    pad1.SetGrid()
    pad1.Draw()
    pad1.cd()

    leg = rt.TLegend(0.6, 0.7, 0.9, 0.9)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    c_out.cd(1)

    for i, h in enumerate(h_list):
        if i == 0:
            h.GetXaxis().SetLabelSize(0)
            h.GetXaxis().SetTitle("")
            # h.GetYaxis().SetRangeUser(0, 1.05*max(map(lambda x: x.GetMaximum(), h_list)))
            h.GetYaxis().SetTitleOffset(1.1)
            h.GetYaxis().SetTitleSize(0.05)
            h.GetYaxis().SetLabelSize(0.05)
            h.SetTitle(title)
            h.DrawCopy(draw_opt)
        else:
            h.DrawCopy(draw_opt+"same")

        leg.AddEntry(h, tag[i], "lep")

    leg.Draw("same")

    c_out.cd()
    pad2 = rt.TPad("pad2", "pad2", 0, 0, 1, 0.3)
    pad2.SetTopMargin(0.03)
    pad2.SetBottomMargin(0.27)
    pad2.SetLeftMargin(0.13)
    pad2.SetGrid()
    pad2.Draw()
    pad2.cd()

    c_out.h_ratio_list = []
    c_out.teff_list = []
    for i, h in enumerate(h_list):
        if i == 0:
            continue
        else:
            h_aux = h.Clone('h_aux'+str(i))
            h_aux.Add(h, h_list[0])

            teff = rt.TEfficiency(h, h_aux)
            teff.SetStatisticOption(rt.TEfficiency.kFCP)
            teff.SetLineColor(h.GetLineColor())
            teff.SetLineWidth(h.GetLineWidth())
            teff.SetTitle(' ;'+h_list_in[0].GetXaxis().GetTitle()+';#varepsilon w/ {};'.format(tag[0]))

            if i == 1:
                teff.Draw('A'+draw_opt)

                rt.gPad.Update()
                graph = teff.GetPaintedGraph()
                graph.GetYaxis().SetTitleOffset(0.5)
                if not ratio_bounds[0] == None:
                    graph.GetHistogram().SetMinimum(ratio_bounds[0])
                if not ratio_bounds[1] == None:
                    graph.GetHistogram().SetMaximum(ratio_bounds[1])

                w = h.GetBinWidth(1)*0.5
                graph.GetXaxis().SetLimits(h.GetBinCenter(1)-w, h.GetBinCenter(h.GetNbinsX())+w)

                graph.GetYaxis().SetTitleSize(0.12)
                graph.GetYaxis().SetLabelSize(0.12)
                graph.GetYaxis().SetNdivisions(506)

                graph.GetXaxis().SetNdivisions(506)
                graph.GetXaxis().SetTitleOffset(0.95)
                graph.GetXaxis().SetTitleSize(0.12)
                graph.GetXaxis().SetLabelSize(0.12)
                graph.GetXaxis().SetTickSize(0.07)

            else:
                teff.Draw(draw_opt)

        c_out.h_ratio_list.append(h)
        c_out.teff_list.append(teff)

    pad2.Update()

    c_out.pad1 = pad1
    c_out.pad2 = pad2
    c_out.h_list = h_list
    c_out.leg = leg

    return c_out
