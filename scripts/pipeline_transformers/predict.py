#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run predictions using the full DrQA retriever-reader pipeline."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import time
import json
import argparse
import logging
from tqdm import tqdm

from drqa import pipeline


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
    parser.add_argument('--retriever-model', type=str, default=None,
                        help="Path to Document Retriever model (tfidf)")
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--n-docs', type=int, default=5,
                        help="Number of docs to retrieve per query")
    parser.add_argument('--group-length', type=int, default=500,
                        help='Target size for squashing short paragraphs together')
    parser.add_argument('--top-n', type=int, default=1,
                        help="Number of predictions to make per query")
    parser.add_argument('--no-cuda', action='store_true',
                        help="Use CPU only")
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help="Specify GPU device id to use")
    # parser.add_argument('--parallel', action='store_true',
    #                     help='Use data parallel (split across gpus)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Document paragraph batching size')
    # parser.add_argument('--predict-batch-size', type=int, default=1000,
    #                     help='Question batching size')
    args = parser.parse_args()
    t0 = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')


    logger.info('Initializing pipeline...')
    DrQA = pipeline.DrQATransformers(
        reader_model=args.reader_model,
        use_fast_tokenizer=args.use_fast_tokenizer,
        group_length=args.group_length,
        cuda=args.cuda,
        ranker_config={
            'options': {
                'tfidf_path': args.retriever_model,
                'strict': False
            }
        },
        db_config={'options': {'db_path': args.doc_db}},
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
        
        progess_bar = tqdm(enumerate(queries), total=len(queries))
        for i, query in progess_bar:
            predictions = DrQA.process(
                query,
                n_docs=args.n_docs,
                top_n=args.top_n,
            )
            f.write(json.dumps(predictions) + '\n')

    logger.info('Total time: %.2f' % (time.time() - t0))
