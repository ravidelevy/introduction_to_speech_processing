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
import pickle


class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """
    CLASSICAL: int = 0
    HEAVY_ROCK: int = 1
    REGGAE: int = 2


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
    train_json_path: str = "jsons/train.json"  # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json"  # you should use this file path to load your test data
    # other training hyperparameters
    weight_scale: int = 0.01
    sample_rate: int = 22050


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyperparameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.00005
    regulariatzion: float = 0.1


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters, **kwargs):
        """
        This defines the classifier object.
        - You should defiend your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization, and you are welcome to experiment
        """
        weight_scale = kwargs['weight_scale'] if 'weight_scale' in kwargs.keys() else 0.001
        input_dim = kwargs['input_dim'] if 'input_dim' in kwargs.keys() else 40
        weights = kwargs['weights'] if 'weights' in kwargs.keys() else None
        biases = kwargs['biases'] if 'biases' in kwargs.keys() else None
        sample_rate = kwargs['sample_rate'] if 'sample_rate' in kwargs.keys() else 22050

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
            mean_matrix = mfcc.mean(1)
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
        # Hinge loss
        y_train = labels.numpy()
        correct_scores = output_scores.numpy()[range(output_scores.shape[0]), y_train]
        margines = np.maximum(output_scores.T - correct_scores + 1, 0).numpy().T
        margines[range(margines.shape[0]), y_train] = 0
        loss = np.sum(margines[margines > 0]) / feats.shape[0] + \
               self.opt_params.regulariatzion * np.sum(np.square(self.weights)) / 2

        gradient = np.zeros(margines.shape)
        gradient[margines > 0] = 1
        gradient[range(gradient.shape[0]), y_train] = -np.sum(gradient, axis=1)

        self.weights -= self.opt_params.learning_rate * (feats.numpy().T.dot(gradient) / feats.shape[0] +
                                                         np.sum(self.weights))
        self.biases -= self.opt_params.learning_rate * gradient.sum(axis=0)

        return loss

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object,
        should return a tuple: (weights, biases)
        """
        return self.weights, self.biases

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor)
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        classification = torch.tensor(np.argmax(self.exctract_feats(wavs).numpy().dot(self.weights) + self.biases,
                                                axis=1))
        classification = classification.reshape(classification.size()[0], 1)
        return classification


class ClassifierHandler:

    checkpoint_file_path = 'model_files\\checkpoints.pkl'

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        train = None
        with open(training_parameters.train_json_path) as train_json:
            train = json.load(train_json)

        sample_rate = sf.read(train[0]['path'])[1]
        mfcc = librosa.feature.mfcc(y=np.array(sf.read(train[0]['path'])[0]), sr=sample_rate, n_mfcc=40)
        music_classifier = MusicClassifier(opt_params=OptimizationParameters(),
                                           input_dim=mfcc.mean(1).size,
                                           sample_rate=sample_rate,
                                           weight_scale=training_parameters.weight_scale)
        random.shuffle(train)
        for epoch in range(training_parameters.num_epochs):
            loss = 0
            for i in range(0, len(train), training_parameters.batch_size):
                batch = train[i:i + training_parameters.batch_size]
                wavs = [sf.read(sample['path'])[0] for sample in batch]
                X_train = music_classifier.exctract_feats(torch.tensor(np.array(wavs)))
                labels = [Genre[sample['label'].upper().replace('-', '_')] for sample in batch]
                y_train = [label.value for label in labels]
                scores = music_classifier.forward(X_train)
                loss += music_classifier.backward(X_train, torch.tensor(scores), torch.tensor(y_train))

            loss /= int(len(train) / training_parameters.batch_size)
            print(f'epoch: {epoch + 1}/{training_parameters.num_epochs}, loss: {loss}')
        
        # Write dictionary pkl file
        weights, biases = music_classifier.get_weights_and_biases()
        ckpt_dict = {'weights': weights, 'biases': biases}
        with open(ClassifierHandler.checkpoint_file_path, 'wb') as fp:
            pickle.dump(ckpt_dict, fp)

        return music_classifier

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights /
        hyperparameters and return the loaded model
        """
        weights, biases = None, None
        music_classifier = None
        # Read dictionary pkl file
        with open(ClassifierHandler.checkpoint_file_path, 'rb') as fp:
            ckpt = pickle.load(fp)
            weights, biases = ckpt['weights'], ckpt['biases']
            music_classifier = MusicClassifier(OptimizationParameters(),
                                               weights=weights, biases=biases)

        return music_classifier
