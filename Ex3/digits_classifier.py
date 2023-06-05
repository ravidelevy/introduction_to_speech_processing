from abc import abstractmethod
import torch
import typing as tp
from dataclasses import dataclass, field
import librosa
import pickle
import os
import numpy as np


@dataclass
class ClassifierArgs:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    # we will use this to give an absolute path to the data, make sure you read the data using this argument. 
    # you may assume the train data is the same
    path_to_training_data_dir: str = "./train_files" 
    
    # you may add other args here
    path_to_model_object: str = "./model.pkl"
    digits: list[str] = field(default_factory=lambda: ['one', 'two', 'three', 'four', 'five'])

class DigitClassifier():
    """
    You should Implement your classifier object here
    """
    def __init__(self, args: ClassifierArgs):
        self.path_to_training_data = args.path_to_training_data_dir
        self.path_to_model_object = args.path_to_model_object
        self.digits = args.digits

        self.features = None
        if os.path.exists(self.path_to_model_object):
            with open(self.path_to_model_object, 'rb') as fp:
                self.features = pickle.load(fp)
        else:
            self.features = self.compute_model_features()
            with open(self.path_to_model_object, 'wb') as fp:
                pickle.dump(self.features, fp)
    
    def get_mfcc_features(self, audio, sr=22050) -> torch.Tensor:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_mels=40,
                                     fmin=0, fmax=None, htk=False)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        return torch.tensor(mfccs_features.T)
    
    def compute_model_features(self) -> torch.Tensor:
        features = []
        for digit in range(len(self.digits)):
            directory = f'{self.path_to_training_data}/{self.digits[digit]}'
            audio_files = [f for f in os.listdir(directory) if not f.startswith('.')]
            digit_features = []
            for file in audio_files:
                audio, sr = librosa.load(f'{directory}/{file}', sr=None)
                digit_features.append(self.get_mfcc_features(audio, sr))
            
            features.append(torch.stack((digit_features)))
        
        return torch.stack((features))

    def extract_features(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> torch.Tensor:
        audio_features = []
        try:
            for path in audio_files:
                audio, sr = librosa.load(path, sr=None)
                audio_features.append(self.get_mfcc_features(audio, sr))
            
        except Exception as e:
            for i in range(audio_files.size()[0]):
                audio_features.append(self.get_mfcc_features(audio[i], sr))
            
            return torch.stack((audio_features))
        
        return torch.stack((audio_features))
    
    def compute_euclidean_distance(self, train_features, test_features)-> torch.Tensor:
        difference = train_features - test_features
        squared_difference = difference ** 2
        sum_of_squared_difference = squared_difference.sum()
        return sum_of_squared_difference.sqrt()

    def compute_dtw(self, train_features, test_features) -> torch.Tensor:
        length = test_features.size()[0]
        distances = []
        for i in range(length):
            distances_per_frame = []
            for j in range(length):
                distances_per_frame.append(((train_features[i] - test_features[j]) ** 2).sum().sqrt())
            
            distances.append(distances_per_frame)
        
        for i in range(1, length):
            distances[i][0] += distances[i - 1][0]

        for j in range(1, length):
            distances[0][j] += distances[0][j - 1]

        for i in range(1, length):
            for j in range(1, length):
                distances[i][j] += min(distances[i - 1][j - 1],
                                       min(distances[i - 1][j],
                                           distances[i][j - 1]))
        
        return distances[-1][-1]

    @abstractmethod
    def classify_using_eucledian_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        test_features = self.extract_features(audio_files)
        labels = []
        for test_sample in range(test_features.size()[0]):
            features_distances = []
            for digit in range(self.features.size()[0]):
                digit_distances = []
                for train_sample in range(self.features.size()[1]):
                    digit_distances.append(self.compute_euclidean_distance(
                                                self.features[digit][train_sample],
                                                test_features[test_sample]))
                
                features_distances.append(digit_distances)
            
            labels.append(torch.argmin(torch.tensor(features_distances)) // self.features.size()[1] + 1)
        
        return labels
    
    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        test_features = self.extract_features(audio_files)
        labels = []
        for test_sample in range(test_features.size()[0]):
            features_distances = []
            for digit in range(self.features.size()[0]):
                digit_distances = []
                for train_sample in range(self.features.size()[1]):
                    digit_distances.append(self.compute_dtw(self.features[digit][train_sample],
                                                            test_features[test_sample]))
                
                features_distances.append(digit_distances)
            
            labels.append(torch.argmin(torch.tensor(features_distances)) // self.features.size()[1] + 1)
        
        return labels

    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using euclidean distance
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        euclidean_labels = self.classify_using_eucledian_distance(audio_files)
        dtw_labels = self.classify_using_DTW_distance(audio_files)

        labeled_filenames = []
        for i in range(len(audio_files)):
            filename = os.path.basename(os.path.normpath(audio_files[i]))
            labeled_filenames.append(f'{filename}-{euclidean_labels[i]}-{dtw_labels[i]}')
        
        with open('output.txt', 'w+') as fp:
            for file in labeled_filenames:
                fp.write(f'{file}\n')
        
        return labeled_filenames
    

class ClassifierHandler:

    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications
        """
        classifier_args = ClassifierArgs()
        return DigitClassifier(classifier_args)

