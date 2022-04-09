#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# thêm chức năng hệ điều hành
import os 
import sys
# from pathlib import PosixPath
# sys lienquan đến hệ thông chương trình
if sys.version_info < (3, 5):
    raise RuntimeError('DrQA supports Python 3.5 or higher.')

DATA_DIR = (
    os.getenv('DRQA_DATA') or
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
)
#nối đường dẫn(lấy tên đường dẫn (đường dẫn))

from . import tokenizers
from . import reader
from . import retriever
from . import pipeline

