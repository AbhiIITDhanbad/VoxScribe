import os
import sys

from SpeechToText.exceptions import STTException
from SpeechToText.logger import logging


class VectorizeChar:

  def __init__(self, max_len = 50):
    try:
        logging.info("Initialize and call the vectorizer class of data utils")
        self.vocab = (
            ["-","#","<",">"]
            + [chr(i+96) for i in range(1,27)]
            + [" ",".",",","?"]
        )

        self.max_len = max_len
        self.char_to_idx = {}

        for i,ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i
    except Exception as e:
        raise STTException(e,sys)

  def __call__(self, text):
    try:
        text = text.lower()
        text = text[:self.max_len - 2]
        text = "<" + text + ">"

        pad_len = self.max_len - len(text)

        return [self.char_to_idx.get(ch,1) for ch in text] + [0] * pad_len
    except Exception as e:
        raise STTException(e,sys)

  def get_vocabulary(self):
    try:
        return self.vocab
    except Exception as e:
        raise STTException(e,sys)


def get_data(wavs, id_to_text, maxlen=50):
    try:
        logging.info("Entered get_data function")
        data = []
        for w in wavs:
            id = os.path.splitext(os.path.basename(w))[0].split(" (1)")[0]
            if len(id_to_text[id]) < maxlen:
                data.append({"audio":w, "text": id_to_text[id]})
        return data
    except Exception as e:
        raise STTException(e,sys)