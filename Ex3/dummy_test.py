from digits_classifier import *
import torch

if __name__ == "__main__":

    handler = ClassifierHandler()
    digits_classifier = handler.get_pretrained_model()

    test_directory = os.path.abspath("test_files")
    test_files = [f for f in os.listdir(test_directory) if not f.startswith('.')]
    test_files = [f'{test_directory}/{file}' for file in test_files]

    classified_files = digits_classifier.classify(test_files)
    euclidean_predictions = torch.tensor([int(prediction.split('-')[-2]) for prediction in classified_files])
    dtw_predictions = torch.tensor([int(prediction.split('-')[-1]) for prediction in classified_files])
    
    similar = (euclidean_predictions == dtw_predictions).float().sum()
    similarity = 100 * similar / len(classified_files)
    print("Similarity = {}".format(similarity))
