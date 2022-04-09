import torch
import regex
import logging
import numpy as np
from math import ceil

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from ..retriever import TfidfDocRanker
from ..retriever import DocDB

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULTS = {
    'ranker': TfidfDocRanker,
    'db': DocDB,
    'reader_model': 'distilbert-base-cased-distilled-squad',
}

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Multiprocessing functions to fetch text
# ------------------------------------------------------------------------------

PROCESS_DB = None

def init(db_class, db_opts):
    global PROCESS_DB
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


# ------------------------------------------------------------------------------
# Main DrQA pipeline
# ------------------------------------------------------------------------------


class DrQATransformers(object):
    def __init__(
        self,
        reader_model=None,
        use_fast_tokenizer=True,
        group_length=500,
        batch_size=32,
        cuda=True,
        num_workers=None,
        db_config=None,
        ranker_config=None,
    ):
        """Initialize the pipeline.

        Args:
            reader_model: name of the Huggingface transformer QA model.
            use_fast_tokenizer: whether to use fast tokenizer
            batch_size: batch size when processing paragraphs.
            cuda: whether to use gpu.
            num_workers: number of parallel CPU processes to use for retrieving
            db_config: config for doc db.
            ranker_config: config for ranker.
            group_length: target size for squashing short paragraphs together.
                0 = read every paragraph independently
                infty = read all paragraphs together
        """
        assert use_fast_tokenizer == True, 'Current version only support models with fast tokenizer'
        self.batch_size = batch_size
        self.device = 'cuda' if cuda else 'cpu'
        self.num_workers = num_workers
        self.group_length = group_length

        logger.info('Initializing document ranker...')
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get('class', DEFAULTS['ranker'])
        ranker_opts = ranker_config.get('options', {})
        self.ranker = ranker_class(**ranker_opts)

        logger.info('Initializing document reader & tokenizer...')
        reader_model = reader_model or DEFAULTS['reader_model']
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

        logger.info('Initializing document retrievers...')
        db_config = db_config or {}
        db_class = db_config.get('class', DEFAULTS['db'])
        db_opts = db_config.get('options', {})

        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(db_class, db_opts)
        )

    def _split_doc(self, doc):
        """Given a doc, split it into chunks (by paragraph)."""
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            curr.append(split)
            curr_len += len(split)
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len > self.group_length:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
        if len(curr) > 0:
            yield ' '.join(curr)

    def process(self, query, top_n=1, n_docs=5, return_context=False):
        """Run a single query."""

        # Rank documents for query.
        ranked = [self.ranker.closest_docs(query, k=n_docs)]
        all_docids, all_doc_scores = zip(*ranked)

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_docids = list({d for docids in all_docids for d in docids})
        # did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        didx2did = {didx: did for didx, did in enumerate(flat_docids)}
        doc_texts = self.processes.map(fetch_text, flat_docids)

        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        flat_splits = []
        didx2sidx = []
        sidx2didx = []
        for i, text in enumerate(doc_texts):
            splits = self._split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
                sidx2didx.append(i)
            didx2sidx[-1][1] = len(flat_splits)

        # Tokenize
        inputs = self.tokenizer(
            [query]*len(flat_splits),
            flat_splits,
            padding=True,
            truncation='only_second',
            stride=96,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=self.need_token_type,
            return_offsets_mapping=True,
            return_tensors='pt',
        ).to(self.device)
        
        # Split batches
        n_examples = inputs.input_ids.shape[0]
        n_batch = ceil(n_examples / self.batch_size)
        batches = [i*self.batch_size for i in range(n_batch)] #[0,32,64,..]
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

        # decode start-end logits to span slice & score
        start, end, score, idx_sort = self.decode_logits(start_logits, end_logits, topk=top_n)

        # Produce predictions, take top_n predictions with highest score
        all_predictions = []
        for i in range(top_n):
            split_id = inputs.overflow_to_sample_mapping[idx_sort[i]].item()
            start_char = inputs.offset_mapping[idx_sort[i], start[i], 0].item()
            end_char = inputs.offset_mapping[idx_sort[i], end[i], 1].item()
            prediction = {
                'doc_id': didx2did[sidx2didx[split_id]],
                'span': flat_splits[split_id][start_char:end_char],
                'doc_score': float(all_doc_scores[0][sidx2didx[split_id]]),
                'span_score': float(score[i]),
            }
            if return_context:
                prediction['context'] = {
                    'text': flat_splits[split_id],
                    'start': start_char,
                    'end': end_char,
                }
            all_predictions.append(prediction)

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
