import numpy as np
import ROOT as rt
import root_numpy as rtnp
import uuid


object_cache = []

hist_style = {
    "LineWidth" : 2,
    "Stats" : 0,
    "Sumw2" : None
}

pad_style = {
    "Grid": None,
    "Logy": None,
}

legend_style = {
    "BorderSize" : 0,
    "FillStyle" : 0
}

line_style = {
    "LineWidth" : 2,
    "LineStyle" : 7
}

style_dict = {
    
    "TH1D" : hist_style,
    "TPad" : pad_style,
    "TCanvas": {},
    "TLegend" : legend_style,
    "TLine" : line_style
}

def apply_properties(obj, props):
    for name, value in props.items():
        # determine the setter to invoke
        setter = getattr(obj, "Set{}".format(name), getattr(obj, name, None))
        #print('setting {} with val {}'.format(setter.__name__, value))
        if value is None:
            setter()
        else:
            setter(value)

def set_style(obj, props={}):
    apply_properties(obj, props)


def create_random_name(prefix="", l=8):
    """
    Creates and returns a random name string consisting of *l* characters using uuid4 internally.
    When *prefix* is given, the name will have the format ``<prefix>_<random_name>``.
    """
    name = uuid.uuid4().hex[:l]
    if prefix:
        name = "{}_{}".format(prefix, name)
    return name


def create_object(cls_name, *args, **kwargs):
    """
    Creates and returns a new ROOT object, constructed via ``ROOT.<cls_name>(*args, **kwargs)`` and
    puts it in an object cache to prevent it from going out-of-scope given ROOTs memory management.
    """
    obj_name = create_random_name(cls_name)
    if cls_name == "TLegend" or cls_name == "TLine": # TLegend has no name argument
        obj = getattr(rt, cls_name)(*args, **kwargs)
    else:    
        obj = getattr(rt, cls_name)(obj_name, *args, **kwargs)
    # set default style
    set_style(obj, props=style_dict[cls_name])
    object_cache.append(obj)
    return obj

def clone_object(obj):
    ''' clones obj and writes clone to cache '''
    clone_name = create_random_name()
    obj_clone = obj.Clone(clone_name)
    set_style(obj)
    object_cache.append(obj_clone)
    return obj_clone

def create_hist(data, title, n_bins, min_bin, max_bin, props):
    h = create_object("TH1D", title, n_bins, min_bin, max_bin)
    rtnp.fill_hist(h, data)
    set_style(h, props=props)
    h.GetXaxis().SetTitleSize(0.1)
    h.GetYaxis().SetTitleSize(0.1)
    #h.GetXaxis().SetLabelSize(15)
    #h.GetYaxis().SetLabelSize(15)
    return h

def create_ratio_hist(h2, h1, target_value=1.):
    ''' creates histogram of h2 / h1 '''
    h3 = clone_object(h2)
    h3.Sumw2()
    h3.Divide(h1)
    line = create_object("TLine", h3.GetXaxis().GetXmin(), target_value, h3.GetXaxis().GetXmax(), target_value)
    return h3, line

def create_canvas_pads():
    canv = create_object("TCanvas","canvas", 600, 600)
    pad1 = create_object("TPad", "pad1", 0, 0.3, 1, 1.0)
    pad1.Draw()
    canv.cd()
    pad2 = create_object("TPad", "pad2", 0, 0.05, 1, 0.3)
    pad2.Draw()
    return canv, pad1, pad2


def make_bg_vs_sig_ratio_plot(mjj_bg_like, mjj_sig_like, target_value, n_bins=50, title="ratio plot", fig_dir=None):
    min_bin = min(np.min(mjj_bg_like), np.min(mjj_sig_like))
    max_bin = max(np.max(mjj_bg_like), np.max(mjj_sig_like))
    print("min {}, max {}".format(min_bin, max_bin))
    # create H1 BG hist
    h1 = create_hist(mjj_bg_like, title, n_bins, min_bin, max_bin, props={"LineColor": rt.kBlue+1, "YTitle": 'num events', "XTitle": "M_{jj} [GeV]"})
    # create H2 SIG hist
    h2 = create_hist(mjj_sig_like, "h2", n_bins, min_bin, max_bin, props={"LineColor": rt.kRed})
    # create H3 RATIO hist
    h3, line = create_ratio_hist(h2, h1, target_value)
    set_style(h3, props={"LineColor": rt.kMagenta+3, "Title": '', "XTitle": 'M_{jj} [GeV]', "YTitle": "ratio SIG / BG"})
    set_style(line, props={"LineColor" : rt.kGreen})
    canv, pad1, pad2 = create_canvas_pads()
    legend = create_object("TLegend", 0.6, 0.7, 0.9, 0.9)
    set_style(legend)
    legend.AddEntry(h1, "BG like")
    legend.AddEntry(h2, "SIG like")
    pad1.cd()
    h1.Draw()
    h2.Draw("Same")
    legend.Draw()
    pad2.cd()
    h3.Draw("ep")
    line.Draw()
    canv.Draw()
    if fig_dir is not None:
        c.SaveAs(os.path.join(fig_dir,title.replace(' ','_')+'.jpg'))
    return


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
