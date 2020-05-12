import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pass', dest='argspass', action='store_true')
    parser.set_defaults(argspass = False)
    parser.add_argument('--sample_rate', dest = 'sr', type = int, default = 16000, help = 'in Hz')
    parser.add_argument('--window', dest = 'window', type = float, default = 25, help = 'in msec')
    parser.add_argument('--overlap', dest='overlap', type = float, default = 10, help = 'in msec')
    args = parser.parse_args()
    return args