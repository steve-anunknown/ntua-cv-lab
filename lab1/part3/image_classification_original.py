import os
from image_classification_original_utils import detector_descriptor
from image_classification_original_utils import detector_names
from image_classification_original_utils import descriptor_names
from image_classification_original_utils import detector_descriptor_names
from image_classification_original_utils import myFastBagOfVisualWords
from cv23_lab1_part3_utils import FeatureExtraction
from cv23_lab1_part3_utils import createTrainTest
from cv23_lab1_part3_utils import svm

if __name__ == '__main__':
    functions_and_files = list(zip(detector_descriptor, files))
    if os.listdir("./features") == []:
        files = ["./features/"+detector+'_'+descriptor+".txt"
                for detector in detector_names
                for descriptor in descriptor_names]
        for name in files:
            os.mknod(name)
        # this takes around 20 minutes
        features_list = [FeatureExtraction(detector, descriptor, saveFile=store)
                        for ((detector, descriptor), store) in functions_and_files]
    else:
        # this is instant provided that the files exist
        # otherwise it fails
        features_list = [FeatureExtraction(detector, descriptor, loadFile=store) 
                        for ((detector, descriptor), store) in functions_and_files]
    
    # this takes close to an hour
    # perhaps the precompiled version uses less clusters and 
    # less max_iterations? I don't know.
    results = []
    for index, feats in enumerate(features_list):
        accs = []
        for k in range(5):
            # Split into a testing set and a test set.
            data_train, label_train, data_test, label_test = createTrainTest(feats, k)

            # Perform Kmeans to find centroids for clusters.
            BOF_tr, BOF_ts = myFastBagOfVisualWords(data_train, data_test, clusters=1000, maxiter=300)
            # Perform Kmeans to find centroids for clusters.
            #BOF_tr, BOF_ts = BagOfWords(data_train, data_test)
            print(BOF_tr.shape, BOF_ts.shape)

            # Train an svm on the training set and make predictions on the test set
            acc, preds, probas = svm(BOF_tr, label_train, BOF_ts, label_test)
            accs.append(acc)
        detector, descriptor = detector_descriptor_names[index]
        results.append(f"{detector}, {descriptor} :\t{100.0*np.mean(accs):.3f}\n")
    with open("mean_acc_mybovw_1000_results.txt", 'w') as file:
        for result in results:
            file.write(result)
    
    # This is perhaps a little bit faster. It takes
    # around 30 minutes instead of 60. The number of iterations
    # do not seem to make it much better. Increasing
    # the tolerance makes it a bit faster, but the
    # accuracy is also slightly decreased.
    results = []
    for index, feats in enumerate(features_list):
        accs = []
        for k in range(5):
            # Split into a testing set and a test set.
            data_train, label_train, data_test, label_test = createTrainTest(feats, k)

            # Perform Kmeans to find centroids for clusters.
            BOF_tr, BOF_ts = myFastBagOfVisualWords(data_train, data_test, clusters=500, maxiter=100)
            # Perform Kmeans to find centroids for clusters.
            #BOF_tr, BOF_ts = BagOfWords(data_train, data_test)
            #print(BOF_tr.shape, BOF_ts.shape)

            # Train an svm on the training set and make predictions on the test set
            acc, preds, probas = svm(BOF_tr, label_train, BOF_ts, label_test)
            accs.append(acc)
        detector, descriptor = detector_descriptor_names[index]
        results.append(f"{detector}, {descriptor} :\t{100.0*np.mean(accs):.3f}\n")
    with open("mean_acc_mybovw_500_results.txt", 'w') as file:
        for result in results:
            file.write(result)