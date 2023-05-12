from abc import abstractmethod
import itertools
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
import json
import random
import soundfile as sf
import librosa
import numpy as np

class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """
    CLASSICAL: int=0
    HEAVY_ROCK: int=1
    REGGAE: int=2


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    batch_size: int = 32
    num_epochs: int = 100
    train_json_path: str = "jsons/train.json" # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json" # you should use this file path to load your test data
    # other training hyper parameters
    weight_scale: int = 0.001
    sample_rate: int = 22050


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.001


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters, **kwargs):
        """
        This defines the classifier object.
        - You should defiend your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment
        """
        weight_scale = kwargs['weight_scale'] if 'weight_scale' in kwargs.keys() else 0.001
        input_dim = kwargs['input_dim'] if 'input_dim' in kwargs.keys() else 520
        weights = kwargs['weights'] if 'weights' in kwargs.keys() else None
        biases = kwargs['biases'] if 'biases' in kwargs.keys() else None
        sample_rate = kwargs['biases'] if 'biases' in kwargs.keys() else 22050

        self.opt_params = opt_params
        self.sample_rate = sample_rate
        self.weights = weight_scale * np.random.randn(input_dim, len(Genre)) if weights is None else weights
        self.biases = np.zeros(len(Genre)) if biases is None else biases

    
    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        features = []
        for wav in wavs:
            mfcc = librosa.feature.mfcc(y=np.array(wav), sr=self.sample_rate, n_mfcc=40)
            mean_matrix = mfcc.mean(0)
            features.append(mean_matrix)
        
        return torch.tensor(np.array(features))

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        return feats.numpy().dot(self.weights) + self.biases
    
    def backward(self, feats: torch.Tensor, output_scores: torch.Tensor, labels: torch.Tensor):
        """
        this function should perform a backward pass through the model.
        - calculate loss
        - calculate gradients
        - update gradients using SGD

        Note: in practice - the optimization process is usually external to the model.
        We thought it may result in less coding needed if you are to apply it here, hence 
        OptimizationParameters are passed to the initialization function
        """
        raise NotImplementedError("optional, function is not implemented")

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object, 
        should return a tuple: (weights, biases)
        """
        return (self.weights, self.biases)
    
    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor) 
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        return np.argmax(self.exctract_feats(wavs).numpy().dot(self.weights) + self.biases, axis=1)
    

class ClassifierHandler:

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        train, test = None, None       
        with open(training_parameters.train_json_path) as train_json:
            train = json.load(train_json)
        
        sample_rate = sf.read(train[0]['path'])[1]
        mfcc = librosa.feature.mfcc(y=np.array(sf.read(train[0]['path'])[0]), sr=sample_rate, n_mfcc=40)
        music_classifier = MusicClassifier(opt_params=OptimizationParameters(),
                                           input_dim=mfcc.mean(0).size,
                                           sample_rate=sample_rate,
                                           weight_scale=training_parameters.weight_scale)
        random.shuffle(train)
        for epoch in range(training_parameters.num_epochs):    
            for i in range(0, len(train), training_parameters.batch_size):
                batch = train[i:i+training_parameters.batch_size]
                wavs = [sf.read(sample['path'])[0] for sample in batch]
                X_train = music_classifier.exctract_feats(torch.tensor(np.array(wavs)))
                y_train = [Genre[sample['label'].upper().replace('-', '_')] for sample in batch]
                scores = music_classifier.forward(X_train)
                music_classifier.backward(X_train, scores, y_train)

        return music_classifier


    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights / 
        hyperparameters and return the loaded model
        """
        weights, biases = None, None
        # TODO: load weights and biases

        music_classifier = MusicClassifier(OptimizationParameters(),
                                           weights=weights, biases=biases)
        return music_classifier
    