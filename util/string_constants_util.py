import datetime


def make_qr_model_str(run, quantile, sig_id, sig_xsec, strategy_id, date=None):
    date_str = ''
    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    return 'QRmodel_run_{}_qnt_{}_{}_sigx_{}_loss_{}_{}.h5'.format(run, str(int(quantile*100)), sig_id, int(sig_xsec), strategy_id, date)
