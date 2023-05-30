from abc import abstractmethod
import torch
import typing as tp
from dataclasses import dataclass, field
import librosa
import pickle
import os


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
    
    def compute_model_features(self) -> torch.Tensor:
        features = [0] * len(self.digits)
        for digit in range(len(self.digits)):
            directory = f'{self.path_to_training_data}/{self.digits[digit]}'
            audio_files = [f for f in os.listdir(directory) if not f.startswith('.')]
            audio, sr = librosa.load(f'{directory}/{audio_files[0]}', sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr)
            features_average = torch.zeros_like(torch.tensor(mfcc.mean(0)))
            for file in audio_files:
                audio, sr = librosa.load(f'{directory}/{file}', sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr)
                features_average += mfcc.mean(0)
            
            features_average /= len(audio_files)
            features[digit] = features_average
        
        return torch.stack((features))

    def extract_features(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> torch.Tensor:
        audio_features = []
        try:
            for path in audio_files:
                audio, sr = librosa.load(path, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr)
                audio_features.append(torch.tensor(mfcc.mean(0)))
        except Exception as e:
            return audio_files
        
        return torch.stack((audio_features))

    def compute_dtw(self, audio_features, digit) -> torch.Tensor:
        length = audio_features.size()[0]
        digit_fetaures = self.features[digit]

        distances = [[] * length] * length
        for i in range(length):
            for j in range(length):
                distances[i].append((audio_features[i] - digit_fetaures[j]) ** 2)
        
        for i in range(1, length):
            distances[i][0] += distances[i - 1][0]

        for j in range(1, length):
            distances[0][j] += distances[0][j - 1]

        for i in range(1, length):
            for j in range(1, length):
                distances[i][j] += (min(distances[i - 1][j - 1],
                                    min(distances[i - 1][j],
                                        distances[i][j - 1])))
        
        return distances[length - 1][length - 1]

    @abstractmethod
    def classify_using_eucledian_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        audio_features = self.extract_features(audio_files)
        labels = []
        for i in range(audio_features.size()[0]):
            features_distances = []
            for digit in range(len(self.digits)):
                features_distances.append(torch.sqrt(torch.sum(torch.pow(
                    self.features[digit] - audio_features[i], 2))))
            
            labels.append(torch.argmin(torch.tensor(features_distances)) + 1)
        
        return labels
    
    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        audio_features = self.extract_features(audio_files)
        labels = []
        for i in range(audio_features.size()[0]):
            features_distances = []
            for digit in range(len(self.digits)):
                features_distances.append(self.compute_dtw(audio_features[i], digit))
            
            labels.append(torch.argmin(torch.tensor(features_distances)) + 1)

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

