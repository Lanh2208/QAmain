from drqa import pipeline

class PyseriniTransformersService():
    def __init__(self, config, lang):
        self.pipeline = pipeline.PyseriniTransformersQA(
            reader_model=config['reader-model'][lang],
            use_fast_tokenizer=config['use-fast-tokenizer'],
            index_path=config['index-path'][lang],
            index_lan=lang,
            batch_size=config['batch-size'],
            cuda=config['cuda'],
            num_workers=config['num-workers'],
            ranker_config=None,
        )

        self.n_docs_default = 30

    def process(self, question, top_n=1, n_docs=30):
        predictions = self.pipeline.process(
            question, top_n, n_docs, return_context=True
        )
        answers = []
        for i, p in enumerate(predictions, 1):
            answers.append({
                'index': i,
                'span': p['span'],
                'doc_id': p['doc_id'],
                'span_score': '%.5g' % p['span_score'],
                'doc_score': '%.5g' % p['doc_score'],
                'text': p['context']['text'],
                'start': p['context']['start'],
                'end': p['context']['end']
            })
        return answers