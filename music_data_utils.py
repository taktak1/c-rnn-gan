#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103

import os, random, csv
import numpy as np


file_list = {}
file_list['train'] = {}
file_list['validation'] = {}
file_list['test'] = {}

file_list['train']['z'] = 'incorpus.ids'
file_list['train']['x'] = 'outcorpus.ids'
file_list['test']['z'] = 'incorpus.ids'
file_list['test']['x'] = 'outcorpus.ids'
file_list['validation']['z'] = 'incorpus.ids'
file_list['validation']['x'] = 'outcorpus.ids'

parts = {'train', 'test', 'validation'}

num_z_features = 1
num_x_features = 1
num_meta_features = 1

class MusicDataLoader(object):

    def __init__(self, datadir):

        self.pointer = {}
        for part in parts:
            self.pointer[part] = 0

        self.datadir = datadir
        if not datadir is None:
            self.read_data()


    def read_data(self):

        self.data = {}
        for part in parts:
            self.data[part] = []

            with open(os.path.join(self.datadir, file_list[part]['z']), 'r') as fz, \
                 open(os.path.join(self.datadir, file_list[part]['x']), 'r') as fx:
                z_lines = fz.readlines()
                x_lines = fx.readlines()
                for z_line, x_line in zip(z_lines, x_lines):
                    z_words = z_line.split()
                    x_words = x_line.split()
                    z_words = list(map(list, zip(z_words)))
                    x_words = list(map(list, zip(x_words)))

                    tmp = {}
                    tmp['z'] = z_words
                    tmp['x'] = x_words
                    self.data[part].append(tmp)

            random.shuffle(self.data[part])
            self.pointer[part] = 0

        return self.data


    def rewind(self, part='train'):
        self.pointer[part] = 0


    def get_batch(self, batchsize, datalength, part='train'):

        if self.pointer[part] > len(self.data[part]) - batchsize:
            return [None, None, None]

        if self.data[part]:
            batch = self.data[part][self.pointer[part]:self.pointer[part] + batchsize]
            self.pointer[part] += batchsize

            batch_z = np.ndarray(shape=[batchsize, datalength, num_z_features])
            batch_x = np.ndarray(shape=[batchsize, datalength, num_x_features])
            batch_meta = np.ndarray(shape=[batchsize, num_meta_features])

            for s in range(len(batch)):
                begin = 0
                if len(batch[s]['z']) < datalength:
                    for e in range(datalength - len(batch[s]['z'])):
                        batch[s]['z'].append([0])

                if len(batch[s]['x']) < datalength:
                    for e in range(datalength - len(batch[s]['x'])):
                        batch[s]['x'].append([0])

                # print('z len:{}'.format(len(batch[s]['z'])))
                # print('x len:{}'.format(len(batch[s]['x'])))
                # print('begin:{}'.format(begin))
                # print('datalength:{}'.format(datalength))

                batch_z[s, :, :] = batch[s]['z'][begin:begin+datalength]
                batch_x[s, :, :] = batch[s]['x'][begin:begin+datalength]
                batch_meta[s, :] = [0]

            # print('batch_z: {}'.format(batch_z.shape))
            # print('batch_x: {}'.format(batch_x.shape))
            # print('batch_meta: {}'.format(batch_meta.shape))
            return [batch_meta, batch_x, batch_z]
        else:
            raise 'data is not initialized.'


    def get_num_z_features(self):
        return num_z_features

    def get_num_x_features(self):
        return num_x_features

    def get_num_meta_features(self):
        return num_meta_features
