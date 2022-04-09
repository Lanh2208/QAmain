#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run predictions using the full retriever-reader pipeline."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import time
import json
import argparse
import logging
from tqdm import tqdm

from drqa.pipeline.pyserini_transformers import PyseriniTransformersQA


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--out-dir', type=str, default='data/pred',
                        help=("Directory to write prediction file to "
                            "(<dataset>-<model>-pipeline.json)"))
    parser.add_argument('--reader-model', type=str, default=None,
                        help="Name of the Huggingface transformer model")
    parser.add_argument('--use-fast-tokenizer', action='store_true', default=True,
                        help="Whether to use fast tokenizer")
    parser.add_argument('--index-path', type=str, default=None,
                        help='Path to the index used for pyserini module')
    parser.add_argument('--index-lan', type=str, default=None,
                        help='language of the index (en, vi, zh...)')
    parser.add_argument('--n-docs', type=int, default=30,
                        help="Number of docs to retrieve per query")
    parser.add_argument('--top-n', type=int, default=1,
                        help="Number of predictions to make per query")
    parser.add_argument('--no-cuda', action='store_true',
                        help="Use CPU only")
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of CPU processes (for fetching text, etc)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help="Specify GPU device id to use")
    # parser.add_argument('--parallel', action='store_true',
    #                     help='Use data parallel (split across gpus)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Document paragraph batching size')
    parser.add_argument('--predict-batch-size', type=int, default=50,
                        help='Question batching size')
    args = parser.parse_args()
    t0 = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')


    logger.info('Initializing pipeline...')
    
    pipeline = PyseriniTransformersQA(
        reader_model=args.reader_model,
        use_fast_tokenizer=args.use_fast_tokenizer,
        index_path=args.index_path,
        index_lan=args.index_lan,
        cuda=args.cuda,
        ranker_config=None, # not implemented
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )


    # ------------------------------------------------------------------------------
    # Read in dataset and make predictions
    # ------------------------------------------------------------------------------


    logger.info('Loading queries from %s' % args.dataset)
    queries = []
    for line in open(args.dataset):
        data = json.loads(line)
        queries.append(data['question'])

    model = os.path.splitext(os.path.basename(args.reader_model or 'default'))[0]
    basename = os.path.splitext(os.path.basename(args.dataset))[0]
    outfile = os.path.join(args.out_dir, basename + '-' + model + '-pipeline.json')

    logger.info('Writing results to %s' % outfile)
    with open(outfile, 'w') as f:
        batches = []
        for i in range(0, len(queries), args.predict_batch_size):
            batches.append(queries[i: i + args.predict_batch_size])

        progess_bar = tqdm(enumerate(batches), total=len(batches))
        for i, queries in progess_bar:
            predictions = pipeline.process_batch(
                queries,
                args.top_n,
                args.n_docs,
            )
            for p in predictions:
                f.write(json.dumps(p) + '\n')

    logger.info('Total time: %.2f' % (time.time() - t0))
