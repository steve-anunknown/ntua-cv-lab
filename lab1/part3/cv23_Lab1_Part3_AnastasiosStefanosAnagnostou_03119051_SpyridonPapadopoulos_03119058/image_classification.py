import os
import sys
sys.path.append('../part12')
import numpy as np

from cv23_lab1_part3_utils import featuresSURF
from cv23_lab1_part3_utils import featuresHOG
from cv23_lab1_part3_utils import FeatureExtraction
from cv23_lab1_part3_utils import createTrainTest
from cv23_lab1_part3_utils import BagOfWords
from cv23_lab1_part3_utils import svm
from corner_detection_utils import CornerDetection
from corner_detection_utils import HarrisLaplacian
from blob_detection_utils import BlobDetection
from blob_detection_utils import HessianLaplacian
from box_detection_utils import BoxLaplacian

# define descriptors and detectors
# also define their names in order to
# accomodate easier file creation
descriptors = [lambda I, kp: featuresSURF(I, kp), lambda I, kp: featuresHOG(I, kp)]
detectors = [lambda i: CornerDetection(i, 2, 2.5, 0.005, 0.05),
            lambda i: HarrisLaplacian(i, 2, 2.5, 0.005, 0.05, 1.1, 6),
            lambda i: BlobDetection(i, 2, 0.005),
            lambda i: HessianLaplacian(i, 2, 0.005, 1.1, 6),
            lambda i: BoxLaplacian(i, 2, 0.005, 1.1, 6)]
descriptor_names = ["featuresSURF", "featuresHOG"]
detector_names = ["CornerDetection", "HarrisLaplacian", "BlobDetection", "HessianLaplacian", "BoxLaplacian"]
detector_descriptor = [(detector, descriptor)
                    for detector in detectors
                    for descriptor in descriptors]
detector_descriptor_names = [(detector, descriptor)
                            for detector in detector_names
                            for descriptor in descriptor_names]

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
        features_list = [FeatureExtraction(detector, descriptor, loadFile=store) 
                        for ((detector, descriptor), store) in functions_and_files]
    
    results = []
    for index, feats in enumerate(features_list):
        accs = []
        for k in range(5):
            # Split into a training set and a test set.
            data_train, label_train, data_test, label_test = createTrainTest(feats, k)

            # Perform Kmeans to find centroids for clusters.
            BOF_tr, BOF_ts = BagOfWords(data_train, data_test)
            # print(BOF_tr.shape, BOF_ts.shape)

            # Train an svm on the training set and make predictions on the test set
            acc, preds, probas = svm(BOF_tr, label_train, BOF_ts, label_test)
            accs.append(acc)

        detector, descriptor = detector_descriptor_names[index]
        results.append(f"{detector}, {descriptor} :\t{100.0*np.mean(accs):.3f}\n")
        
    with open("mean_acc_bovw_results_test.txt", 'w') as file:
        for result in results:
            file.write(result)
