from drqa import pipeline

class DrQATransformersService():
    def __init__(self, config, lang):
        self.pipeline = pipeline.DrQATransformers(
            reader_model=config['reader-model'][lang],
            use_fast_tokenizer=config['use-fast-tokenizer'],
            group_length=config['group-length'],
            batch_size=config['batch-size'],
            cuda=config['cuda'],
            num_workers=config['num-workers'],
            db_config={'options': {'db_path': config['doc-db'][lang]}},
            ranker_config={
                'options': {
                    'tfidf_path': config['retriever-model'][lang],
                    'strict': False
                }
            },
        )

        self.n_docs_default = 5

    def process(self, question, top_n=1, n_docs=5):
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