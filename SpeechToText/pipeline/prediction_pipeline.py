import os, sys
import tensorflow as tf

from pydub import AudioSegment 
from SpeechToText.models.model import Transformer
from SpeechToText.prediction_utils import path_to_audio
from SpeechToText.models.data_utils import VectorizeChar
from SpeechToText.constants import MAX_TARGET_LENGTH
from SpeechToText.logger import logging
from SpeechToText.exceptions import STTException


class Prediction:
    def __init__(self, audio_path, model_path):
        try:
            self.vectorizer = VectorizeChar(MAX_TARGET_LENGTH)
            self.audio_path = audio_path
            if(str(self.audio_path).endswith(".mp3")):
                audio = AudioSegment.from_mp3(self.audio_path)
                audio.export(self.audio_path, format="wav")
                logging.info("mp3 converted to wav")

            if os.path.isdir(model_path):
                weight_files = [f for f in os.listdir(model_path) if f.endswith('.weights.h5')]
                if weight_files:
                    model_path = os.path.join(model_path, weight_files[0])
                    logging.info(f"Resolved model path to: {model_path}")
                else:
                    logging.warning(f"No .weights.h5 file found in {model_path}")
            self.model_path = model_path
        except Exception as e:
            raise STTException(e, sys)
    
    def prediction(self):
        try:
            idx_to_char = self.vectorizer.get_vocabulary()

            logging.info("vocabulary created")

            model = Transformer(
                num_hid=200,
                num_head=2,
                num_feed_forward=400,
                target_maxlen=MAX_TARGET_LENGTH,
                num_layers_enc=4,
                num_layers_dec=1,
                num_classes=34,
            )

            logging.info("model instance created")


            dummy_source = tf.zeros((1, 2754, 129))
            dummy_target = tf.zeros((1, MAX_TARGET_LENGTH), dtype=tf.int32)
            model([dummy_source, dummy_target])
            model.load_weights(self.model_path)
            logging.info("model weights loaded")

            preds = model.generate(tf.expand_dims(path_to_audio(path=self.audio_path), axis=0), target_start_token_idx=2)

            preds = preds.numpy()

            prediction = ""
            for idx in preds[0]:
                prediction += idx_to_char[idx]
                if idx_to_char[idx] == '>':
                    break
            
            logging.info("Prediction completed")
            
            return str(prediction)
        except Exception as e:
            raise STTException(e, sys)