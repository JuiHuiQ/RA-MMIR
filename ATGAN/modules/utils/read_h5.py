# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : read_h5.py

Author     ：yujing_rao
"""
import h5py
import numpy as np


def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label