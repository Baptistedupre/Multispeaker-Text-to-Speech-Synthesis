from text import cmudict


_pad = '_'
_eos = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? ' # noqa E501

_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + _arpabet


if __name__ == '__main__':
    print(symbols)
