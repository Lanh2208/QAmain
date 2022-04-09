import logging

from CocCocTokenizer import PyTokenizer
from .tokenizer import Tokens, Tokenizer

logger = logging.getLogger(__name__)


class CocCocTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()
        self.tokenizer = PyTokenizer()

    def get_ws(self, tokens):
        length = len(tokens)
        ws = [None] * (length + 1)
        ws[0] = -1

        for i in range(1, length + 1):
            ws[i] = ws[0] + len(tokens[i - 1]) + 1

        return ws

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = self.tokenizer.word_tokenize(clean_text)
        ws = self.get_ws(tokens)

        data = []
        for i in range(len(tokens)):
            # Get text
            token = tokens[i]

            # Get whitespace
            start_ws = ws[i] + 1
            end_ws = ws[i + 1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                (start_ws, end_ws),
            ))
        return Tokens(data, self.annotators)
