import pofah.jet_sample as js
import pofah.util.sample_factory as sf
import pofah.util.result_writer as reswr
import pofah.util.experiment as ex
import selection.discriminator as dis
import selection.loss_strategy as ls
import analysis.analysis_discriminator as an
import anpofah.util.plotting_util as pu
import pofah.path_constants.sample_dict_file_parts_reco as sd 
import datetime
import dadrah.analysis.root_plotting_util as rpu
import dadrah.selection.selection_util as seu
from importlib import reload
import os
import setGPU


# read in qcd signal region sample
run_n = 101
SM_sample = 'qcdSigAllReco'
#BSM_samples = ['GtoWW15naReco', 'GtoWW15brReco', 'GtoWW25naReco', 'GtoWW25brReco','GtoWW35naReco', 'GtoWW35brReco', 'GtoWW45naReco', 'GtoWW45brReco']
BSM_samples = ['GtoWW25naReco', 'GtoWW35naReco']
all_samples = [SM_sample] + BSM_samples
mjj_key = 'mJJ'
reco_loss_j1_key = 'j1RecoLoss'
QR_train_share = 0.3


experiment = ex.Experiment(run_n).setup(analysis_dir=True)
paths = sf.SamplePathDirFactory(sd.path_dict).extend_base_path(experiment.run_dir)

data = sf.read_inputs_to_jet_sample_dict_from_dir(all_samples, paths)

# define quantile and loss-strategy for discimination
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9] # 5%
strategy = ls.combine_loss_min
qcd_sig_sample = data[SM_sample]
#split qcd sample into training and testing
qcd_train, qcd_test = js.split_jet_sample_train_test(qcd_sig_sample, QR_train_share)
# update data_dictionary
data[SM_sample] = qcd_test
print(qcd_sig_sample.features())

for quantile in quantiles:

    discriminator = dis.QRDiscriminator(quantile=quantile, loss_strategy=strategy, n_nodes=70)
    discriminator.fit(qcd_train)

    # plot mjj qcd sig to check for flat ratio
    an.analyze_discriminator_cut(discriminator, qcd_train)


# In[8]:


date = datetime.date.today()
date_str = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
model_str = str(QR_train_share)+'_qr_train_'+ str(quantile) +'q_'+ date_str
discriminator.save('models/dnn_run_101_{}.h5'.format(model_str))
print(model_str)


# ## plot mjj accepted vs rejected signal to check for mass sculpting

# In[9]:


counting_experiment = {}
bin_edges = [0,1126,1181,1246,1313,1383,1455,1530,1607,1687,1770,1856,1945,2037,2132,2231,2332,2438,2546,2659,2775,2895,3019,3147,3279,3416,3558,3704,3854,4010,4171,4337,4509,4686,4869,5058,5253,5500,5663,5877,6099,6328,6564,6808,1e6]


# ### qcd training set

# In[10]:


reload(rpu)
selection = discriminator.select(qcd_train)
qcd_train.add_feature('sel', selection)
title = "QCD training set: BG like vs SIG like mjj distribution and their ratio"
h_bg_like_qcd_train, h_sig_like_qcd_train = rpu.make_bg_vs_sig_ratio_plot(qcd_train.rejected(mjj_key), qcd_train.accepted(mjj_key), target_value=quantile, n_bins=30, title=title, fig_dir=experiment.analysis_dir_fig)


# ### qcd test set

# In[11]:


sample = data[SM_sample]
# apply selection to datasample
selection = discriminator.select(sample)
#qcd_sig_test_selected = discriminator.apply(qcd_sig_test)
sample.add_feature('sel', selection)
title = "QCD test set: BG like vs SIG like mjj distribution and ratio"
h_bg_like_qcd_test, h_sig_like_qcd_test = rpu.make_bg_vs_sig_ratio_plot(sample.rejected(mjj_key), sample.accepted(mjj_key), target_value=quantile, n_bins=30, title=title, fig_dir=experiment.analysis_dir_fig)
# save in counts sig like & bg like for qcd SR test set
counting_experiment[SM_sample] = seu.get_bin_counts_sig_like_bg_like(sample, bin_edges)


# ## check p-value GOF qcd training-set

# In[12]:


import dadrah.statistics.hypothesis_test as hypo
reload(hypo)
hypo.hypothesis_test(h_bg_like_qcd_train, h_sig_like_qcd_train, quantile, N_asymov=10000)


# ## check p-value GOF qcd test-set

# In[13]:


hypo.hypothesis_test(h_bg_like_qcd_test, h_sig_like_qcd_test, quantile, N_asymov=10000)


# # apply selection

# In[14]:


for sample_id in BSM_samples:
    # apply selection to datasample
    selection = discriminator.select(data[sample_id])
    #qcd_sig_test_selected = discriminator.apply(qcd_sig_test)
    data[sample_id].add_feature('sel', selection)


# ## print efficiency table 

# In[62]:


reload(an)
an.print_discriminator_efficiency_table(data)


# # plot mjj ratio

# In[16]:


for sample_id in BSM_samples:
    sample = data[sample_id]
    title = sample.name + ": BG like vs SIG like mjj distribution and ratio"
    rpu.make_bg_vs_sig_ratio_plot(sample.rejected(mjj_key), sample.accepted(mjj_key), target_value=quantile, n_bins=30, title=title, fig_dir=experiment.analysis_dir_fig)
    # save in counts sig like & bg like for qcd SR test set
    counting_experiment[sample_id] = seu.get_bin_counts_sig_like_bg_like(sample, bin_edges)


# ## plot losses (reco vs kl) for accepted and rejected sample

# In[17]:


for sample in data.values():
    # plot BG like J1
    pu.plot_hist_2d(sample.rejected('j1RecoLoss'),sample.rejected('j1KlLoss'), xlabel='reco loss', ylabel='kl loss', title=sample.name+' Reco vs KL J1 BG like', clip_outlier=True, fig_dir=experiment.analysis_dir_fig, plot_name=sample.plot_name()+'_RecoVsKLJ1_BGlike.png')
    # plot SIG like J1
    pu.plot_hist_2d(sample.accepted('j1RecoLoss'),sample.accepted('j1KlLoss'), xlabel='reco loss', ylabel='kl loss', title=sample.name+' Reco vs KL J1 SIG like', clip_outlier=True, fig_dir=experiment.analysis_dir_fig, plot_name=sample.plot_name()+'_RecoVsKLJ1_SIGlike.png')
    # plot BG like J2
    pu.plot_hist_2d(sample.rejected('j2RecoLoss'),sample.rejected('j2KlLoss'), xlabel='reco loss', ylabel='kl loss', title=sample.name+' Reco vs KL J2 BG like', clip_outlier=True, fig_dir=experiment.analysis_dir_fig, plot_name=sample.plot_name()+'_RecoVsKLJ2_BGlike.png')
    # plot SIG like J1
    pu.plot_hist_2d(sample.accepted('j2RecoLoss'),sample.accepted('j2KlLoss'), xlabel='reco loss', ylabel='kl loss', title=sample.name+' Reco vs KL J2 SIG like', clip_outlier=True, fig_dir=experiment.analysis_dir_fig, plot_name=sample.plot_name()+'_RecoVsKLJ2_SIGlike.png')
    


# In[18]:


print(experiment.analysis_dir)


# ## write selected samples

# In[58]:


import pofah.path_constants.sample_dict_file_parts_selected as sds
result_paths = sf.SamplePathDirFactory(sds.path_dict).extend_base_path(experiment.run_dir)


# In[61]:


for sample_id, sample in data.items():
    #print('writing results for {} to {}'.format(sds.path_dict['sample_name'][sample_id], os.path.join(result_paths.sample_dir_path(sample_id), result_paths.sample_file_path(sample_id))))
    sample.dump(os.path.join(result_paths.sample_dir_path(sample_id), result_paths.sample_file_path(sample_id)))


# ## write bin counts to file

# In[19]:


reload(reswr)
import os
reswr.write_bin_counts_to_file(counting_experiment, bin_edges, os.path.join(experiment.analysis_dir_bin_count,'sel_bin_count_'+experiment.run_dir+'_tsz'+str(int(QR_train_share*100))+'pc_q'+str(quantile*100)+'.h5'))


# In[ ]:




