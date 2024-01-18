"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import tqdm
import argparse
import random

import torch

from .model import WaveEncoder, WaveDecoder

from .utils.core import feature_wct
from .utils.io import Timer, open_image, load_segment, compute_label_info


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    def __init__(self, model_path='WCT2/model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device='cuda:0', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], style_skips[skip_level][component],
                                                                       content_segment, style_segment,
                                                                       label_set, label_indicator,
                                                                       alpha=alpha, device=self.device)
                self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret


def style_transfer(content, style, device):

    transfer_at = set()
    transfer_at.add('encoder')
    transfer_at.add('decoder')

    wct2 = WCT2(transfer_at=transfer_at, option_unpool='cat5', device=device, verbose=False)
    batch_size = content.size(0)
    transferred_imgs = []
    for i in range(batch_size):
        with torch.no_grad():
            img = wct2.transfer(content[i,...].unsqueeze(0), style[i,...].unsqueeze(0), content_segment=None, style_segment=None, alpha=1)
            transferred_imgs.append(img.squeeze())

    transferred_img = transferred_imgs[0][0].unsqueeze(0).unsqueeze(0)
    for i in range(1, batch_size):
        transferred_img = torch.cat((transferred_img, transferred_imgs[i][0].unsqueeze(0).unsqueeze(0)), 0)  

    return transferred_img
       