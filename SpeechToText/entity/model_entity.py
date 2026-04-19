import sys
import tensorflow as tf

from SpeechToText.logger import logging
from SpeechToText.exceptions import STTException
from SpeechToText.models.data_utils import VectorizeChar

class CreateTensors:
    def __init__(self, data: list, vectorizer: VectorizeChar) -> None:
        try:
            self.data = data
            self.vectorizer = vectorizer
        except Exception as e:
            raise STTException(e,sys)

    def create_text_ds(self):

        try:
            logging.info("Entered create_text_ds")
            texts = [_["text"] for _ in self.data]
            text_ds = [self.vectorizer(t) for t in texts]
            text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
            logging.info("Exited create_text_ds function")
            return text_ds
        except Exception as e:
            raise STTException(e,sys)

    def path_to_audio(self,path):
        try:
            logging.info("Entered path_to_audio")
            audio = tf.io.read_file(path)
            audio, _ = tf.audio.decode_wav(audio, 1)
            audio = tf.squeeze(audio, axis=-1)
            stfts = tf.signal.stft(audio, frame_length = 200, frame_step = 80, fft_length = 256)

            x = tf.math.pow(tf.abs(stfts),0.5)

            means = tf.math.reduce_mean(x, 1, keepdims=True)
            stddevs = tf.math.reduce_std(x, 1, keepdims=True)

            # x = [x-means] / stddevs
            x = (x-means) / stddevs
            audio_len = tf.shape(x)[0]
            pad_len = 2754
            paddings = tf.constant([[0,pad_len],[0,0]])
            x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
            logging.info("Exited path_to_audio function")
            return x
        except Exception as e:
            raise STTException(e,sys)

    def create_audio_ds(self):
        try:
            logging.info("Entered the create_audio_ds function")
            flist = [_["audio"] for _ in self.data]
            audio_ds = tf.data.Dataset.from_tensor_slices(flist)

            audio_ds = audio_ds.map(
                self.path_to_audio, num_parallel_calls = tf.data.AUTOTUNE

            )
            logging.info("Exited the create_audio_ds function")
            return audio_ds

        except Exception as e:
            raise STTException(e,sys)
            

    def create_tf_dataset(self, bs=4):
        try:
            logging.info("Entered the create_tf_dataset function")
            audio_ds = self.create_audio_ds()
            text_ds = self.create_text_ds()
            # return audio_ds, text_ds
            ds = tf.data.Dataset.zip((audio_ds, text_ds))
            ds = ds.map(lambda x, y:{"source":x, "target":y})
            ds = ds.cache()
            ds = ds.batch(bs)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            logging.info("Exited the create_tf_dataset function ")
            return ds

        except Exception as e:
            raise STTException(e,sys)
    

