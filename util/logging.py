import logging

def get_logger(name):

    logger = logging.getLogger(name)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter('%(asctime)s [%(filename)s:%(funcName)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    return logger
