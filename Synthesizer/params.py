import yaml


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def load_params():
    with open('synthesizer/config/config.yaml', 'r') as f:
        params = yaml.load_all(f, Loader=yaml.FullLoader)
        dict_params = {}
        for doc in params:
            for k, v in doc.items():
                dict_params[k] = v
    return dict_params


hparams = DotDict(load_params())
