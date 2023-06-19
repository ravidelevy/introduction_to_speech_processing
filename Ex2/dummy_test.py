from genre_classifier import *

if __name__ == "__main__":

    handler = ClassifierHandler()

    # check that training is working
    training_params = TrainingParameters(batch_size=32, num_epochs=100)
    music_classifier = None
    try:
        music_classifier = handler.train_new_model(training_params)
        print("Train dummy test passed")
    except Exception as e:
        print(f"Train dummy test failed, exception:\n{e}")

    test = None
    with open(training_params.test_json_path) as test_json:
        test = json.load(test_json)

    wavs = [[sf.read(sample['path'])[0]] for sample in test]
    labels = [Genre[sample['label'].upper().replace('-', '_')] for sample in test]
    y_pred = music_classifier.classify(torch.tensor(np.array(wavs)))
    y_test = torch.tensor([[label.value] for label in labels])

    correct = (y_pred == y_test).float().sum()
    accuracy = 100 * correct / len(test)
    print("Accuracy = {}".format(accuracy))

    # check that model object is obtained
    try:
        music_classifier = handler.get_pretrained_model()
        print("Get pretrained object dummy test passed")
    except Exception as e:
        print(f"Get pretrained object dummy test failed, exception:\n{e}")

    # feel free to add tests here. 
    # We will not be giving score to submitted tests.
    # You may (and recommended to) share tests with one another.
    y_pred = music_classifier.classify(torch.tensor(np.array(wavs)))
    y_test = torch.tensor([[label.value] for label in labels])

    correct = (y_pred == y_test).float().sum()
    accuracy = 100 * correct / len(test)
    print("Accuracy = {}".format(accuracy))
