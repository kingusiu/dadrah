import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats
import dadrah.analysis.root_plotting_util as rpu


def hypothesis_test(h_sig_like, h_bg_like, eff, N_asymov = 10000, max_N_asymov = 1e7, show_hist=True):
    
    h_sig_like_bin_content, _ = rpu.get_bin_counts_positions_from_hist(h_sig_like)
    h_bg_like_bin_content, _ = rpu.get_bin_counts_positions_from_hist(h_bg_like)
    
    nu = eff*h_bg_like_bin_content/(1-eff)

    probs_obs = sp.stats.poisson.pmf(h_sig_like_bin_content.astype(np.int), nu)
    probs_obs = np.where(probs_obs < 1e-10, np.full_like(probs_obs, 1e-10), probs_obs)
    s_obs = np.sum(-np.log(probs_obs), axis=-1)
    print('S obs:', s_obs)

    N_worse = 0
    N_tot = 0
    loops = 0
    # toy generation
    while N_worse < 25 and N_tot < max_N_asymov:
        loops += 1
        if loops > 1 and loops%10 == 0:
            print(N_tot, N_worse)
        if loops == 10:
            print('Increasing by a factor 5 the number of asymov per loop')
            N_asymov *=5
        o_asymov = np.random.poisson(nu, (N_asymov, nu.shape[0]))
        probs = sp.stats.poisson.pmf(o_asymov, nu)
        probs = np.where(probs < 1e-10, np.full_like(probs, 1e-10), probs)
        nll = -np.log(probs)
        s_asymov = np.sum(nll, axis=-1)

        N_worse += np.sum(s_asymov > s_obs) # compare accepted-stats and toy-stats
        N_tot += N_asymov

        if max_N_asymov/N_tot < 25 and (N_worse * (max_N_asymov/N_tot) < 25):
            print('Will never have enough stat - giving up.')
            p_val = max(1, N_worse)/float(N_tot)
            return p_val

    print('Test stat reached after {} loops'.format(loops))

    p_val = max(1, N_worse)/float(N_tot)
    
    if show_hist:
        plt.figure()
        binContent, _, _ = plt.hist(s_asymov, label='Distribution assuming eff={:.1f}%'.format(100*eff))
        plt.plot([s_obs, s_obs], [0,np.max(binContent)], label='Observed')
        plt.legend(loc='best')
        plt.xlabel('Test statistic')
        plt.ylabel('Entries')
        plt.title('hypothesis test QR training qcd, p-val = ' + str(p_val) )
        #plt.savefig(os.path.join(fig_dir,'qcd_'+qr_data+'qr_selection_hist2d.png'))

    return p_val


