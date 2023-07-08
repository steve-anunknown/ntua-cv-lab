import sys
sys.path.append('../part12')
import numpy as np

from cv23_lab1_part3_utils import featuresSURF
from cv23_lab1_part3_utils import featuresHOG
from cv23_lab1_part3_utils import matching_evaluation
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

if __name__ == '__main__':
    # this takes around 1-2 minutes
    avg_scale_errors_list = np.zeros((len(descriptors), len(detectors), 3))
    avg_theta_errors_list = np.zeros((len(descriptors), len(detectors), 3))
    for index, descriptor in enumerate(descriptors):
        for jndex, detector in enumerate(detectors):
            avg_scale_errors_list[index, jndex], avg_theta_errors_list[index, jndex] = matching_evaluation(detector, descriptor)

    np.set_printoptions(precision=3)
    minim = np.mean(avg_scale_errors_list[0,0])
    for index, descriptor in enumerate(descriptor_names):
        for jndex, detector in enumerate(detector_names):
            if minim > np.mean(avg_scale_errors_list[index, jndex]):
                minim = np.mean(avg_scale_errors_list[index, jndex])
                bestpair = (descriptor, detector)
    with open("../report/avg_scale_errors.txt", 'w') as file:
        for index, descriptor in enumerate(descriptor_names):
            for jndex, detector in enumerate(detector_names):
                if (descriptor, detector) == bestpair:
                    file.write(f"{descriptor}, {detector} :\t{avg_scale_errors_list[index, jndex]}\t BEST\n")
                else:
                    file.write(f"{descriptor}, {detector} :\t{avg_scale_errors_list[index, jndex]}\t\n")

    minim = np.mean(avg_theta_errors_list[0,0])
    for index, descriptor in enumerate(descriptor_names):
        for jndex, detector in enumerate(detector_names):
            if minim > np.mean(avg_scale_errors_list[index, jndex]):
                minim = np.mean(avg_scale_errors_list[index, jndex])
                bestpair = (descriptor, detector)
    with open("../report/avg_theta_errors.txt", 'w') as file:
        for index, descriptor in enumerate(descriptor_names):
            for jndex, detector in enumerate(detector_names):
                if (descriptor, detector) == bestpair:
                    file.write(f"{descriptor}, {detector} :\t{avg_theta_errors_list[index, jndex]}\t BEST\n")
                else:
                    file.write(f"{descriptor}, {detector} :\t{avg_theta_errors_list[index, jndex]}\n")