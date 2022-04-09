import torch
import logging
import numpy as np
from math import ceil
import json

from pyserini.search import SimpleSearcher
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULTS = {
    'index_path': 'data/index/lucene-index.enwiki-20180701-paragraphs',
    'index_lan': 'en',
    'reader_model': 'distilbert-base-cased-distilled-squad',
}

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------


class PyseriniTransformersQA(object):
    def __init__(
        self,
        index_path=None,
        index_lan=None,
        reader_model=None,
        use_fast_tokenizer=True,
        batch_size=32,
        cuda=True,
        ranker_config=None, # not implemented
        num_workers=2,
    ):
        """Initialize the pipeline.

        Args:
            reader_model: name or path to Huggingface transformer QA model.
            use_fast_tokenizer: whether to use fast tokenizer
            batch_size: batch size when processing passages.
            cuda: whether to use gpu for reader inference.
            index_path: path to the index used for pyserini module
            index_lan: language of the index ('en', 'vi', 'zh'...)
            ranker_config:  #not implemented (k1, b)
            num_workers: number of parallel CPU processes to use for retrieving
        """
        assert use_fast_tokenizer == True, 'Current version only support models with fast tokenizer'
        self.batch_size = batch_size
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.num_workers = num_workers
        index_path = index_path or DEFAULTS['index_path']
        index_lan = index_lan or DEFAULTS['index_lan']
        reader_model = reader_model or DEFAULTS['reader_model']

        logger.info(f'Initializing document ranker/retriever from index: {index_path}, language: {index_lan}')
        self.retriever = SimpleSearcher(index_path)
        self.retriever.set_bm25(k1=0.9, b=0.4)
        self.retriever.object.setLanguage(index_lan)

        logger.info(f'Initializing document reader & tokenizer from: {reader_model}')
        self.reader = AutoModelForQuestionAnswering \
            .from_pretrained(reader_model) \
            .eval() \
            .to(self.device)
        self.need_token_type = self.reader.base_model_prefix not in {
            "xlm", "roberta", "distilbert", "camembert", "bart", "longformer"
        }
        tokenizer_kwargs = {}
        if self.reader.base_model_prefix in {'mobilebert'}:
            tokenizer_kwargs['model_max_length'] = self.reader.config.max_position_embeddings
        #
        self.tokenizer = AutoTokenizer.from_pretrained(reader_model, use_fast=use_fast_tokenizer, **tokenizer_kwargs)


    def process(self, query, top_n=1, n_passages=30, return_context=False):
        """Run a single query."""
        return self.process_batch([query], top_n, n_passages, return_context)[0]


    def process_batch(self, queries, top_n=1, n_passages=30, return_context=False):
        """Run a batch of queries (more efficient)."""

        queries_ids = [str(i) for i in range(len(queries))]
        query_dict = self.retriever.batch_search(queries, queries_ids, k=n_passages, threads=self.num_workers)
        
        input_queries = []
        passages = []
        passage_ids = []
        passage_scores = []
        for i, (query_id, query_results) in enumerate(query_dict.items()):
            input_queries.extend([queries[i]]*n_passages)
            for passage in query_results:
                try:
                    content = json.loads(passage.raw)['contents']
                except Exception as e:
                    content = passage.raw
                
                passages.append(content)
                passage_ids.append(passage.docid)
                passage_scores.append(passage.score)

        # Tokenize
        inputs = self.tokenizer(
            input_queries,
            passages,
            padding=True,
            truncation='only_second',
            stride=96,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=self.need_token_type,
            return_offsets_mapping=True,
            return_tensors='pt',
        ).to(self.device)
        n_examples = inputs.input_ids.shape[0]

        # Handle truncation mapping
        indexes = torch.where(inputs.overflow_to_sample_mapping % n_passages == 0)[0].tolist()
        indexes.append(n_examples)  # example: [0,30,61,91,123,124,154,...]
        query_offset = [] # example k=30: [0,30,61,91,124,154,...], len=len(queries)+1
        for i in range(0,len(indexes)-1):
            if indexes[i+1] - indexes[i] > 1:  # not work for k=1
                query_offset.append(indexes[i])
        query_offset[0] = 0
        query_offset.append(indexes[-1])
        assert len(query_offset) == len(queries)+1, "Oops a bug, Im sure have tested this query_offset carefully. Issue me"
        
        # Split batches
        n_batch = ceil(n_examples / self.batch_size)
        batches = [i*self.batch_size for i in range(n_batch)] #[0,32,64,..] # or should i name it batch_offset
        batches.append(n_examples)

        # Feed forward batches
        outputs = []
        for i in range(n_batch):
            with torch.no_grad():
                if self.need_token_type:
                    output = self.reader(
                        inputs.input_ids[batches[i]:batches[i+1]],
                        inputs.attention_mask[batches[i]:batches[i+1]],
                        token_type_ids=inputs.token_type_ids[batches[i]:batches[i+1]],
                    )
                else:
                    output = self.reader(
                        inputs.input_ids[batches[i]:batches[i+1]],
                        inputs.attention_mask[batches[i]:batches[i+1]],
                    )
                outputs.append(output)

        # Join batch outputs
        start_logits = torch.cat([o.start_logits for o in outputs], dim=0)
        end_logits = torch.cat([o.end_logits for o in outputs], dim=0)

        # predictions for each query
        all_predictions = []
        for query_id in range(len(queries)):
            query_offset_start = query_offset[query_id]
            query_offset_end = query_offset[query_id+1]
            query_start_logits = start_logits[query_offset_start:query_offset_end]
            query_end_logits = end_logits[query_offset_start:query_offset_end]

            # decode start-end logits to span slice & score
            start, end, score, idx_sort = self.decode_logits(query_start_logits, query_end_logits, topk=top_n)

            # Produce predictions, take top_n predictions with highest score
            query_predictions = []
            for i in range(top_n):
                idx_sort[i] += query_offset_start
                passage_idx = inputs.overflow_to_sample_mapping[idx_sort[i]].item()
                start_char = inputs.offset_mapping[idx_sort[i], start[i], 0].item()
                end_char = inputs.offset_mapping[idx_sort[i], end[i], 1].item()
                prediction = {
                    'doc_id': passage_ids[passage_idx],
                    'span': passages[passage_idx][start_char:end_char],
                    'doc_score': float(passage_scores[passage_idx]),
                    'span_score': float(score[i]),
                }
                if return_context:
                    prediction['context'] = {
                        'text': passages[passage_idx],
                        'start': start_char,
                        'end': end_char,
                    }
                query_predictions.append(prediction)
            all_predictions.append(query_predictions)

        return all_predictions


    def decode_logits(self, start_logits, end_logits, topk=1, max_answer_len=None):
        """
        Take the output of any :obj:`ModelForQuestionAnswering` and generate score for each span to be the actual answer.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
        answer end position being before the starting position. The method supports output the k-best answer through
        the topk argument.

        Args:
            start_logits (:obj:`tensor`): Individual start logits for each token. # shape: batch, len(input_ids[0])
            end_logits (:obj:`tensor`): Individual end logits for each token.
            topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
        Output:
            starts:  top_n predicted start indices
            ends:  top_n predicted end indices
            scores:  top_n prediction scores
            idx_sort:  top_n batch element ids
        """
        start = start_logits.cpu().numpy().clip(min=0.0)
        end = end_logits.cpu().numpy().clip(min=0.0)
        max_answer_len = max_answer_len or start.shape[1]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        idx_sort, starts, ends = np.unravel_index(idx_sort, candidates.shape)
        scores = candidates[idx_sort, starts, ends]

        return starts, ends, scores, idx_sort
