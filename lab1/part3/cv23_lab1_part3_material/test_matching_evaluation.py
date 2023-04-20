# Assuming detectors are in file "cv20_lab1_part2.py", replace with your filename.
import cv20_lab1_part3_utils as p3
import cv20_lab1_part2 as p2

if __name__ == '__main__':
    # Here is a lambda which acts as a wrapper for detector function, e.g. harrisDetector.
    # The detector arguments are, in order: image, sigma, rho, k, threshold.
    detect_fun = lambda I: p2.harrisDetector(I, 2, 2.5, 0.05, 0.005)

    # You can use either of the following lines to extract features (HOG/SURF).
    desc_fun = lambda I, kp: p3.featuresSURF(I,kp)
    # desc_fun = lambda I, kp: p3.featuresHOG(I,kp)

    # Execute evaluation by providing the above functions as arguments
    # Returns 2 1x3 arrays containing the errors
    avg_scale_errors, avg_theta_errors = p3.matching_evaluation(detect_fun, desc_fun)
    print('Avg. Scale Error for Image 1: {:.3f}'.format(avg_scale_errors[0]))
    print('Avg. Theta Error for Image 1: {:.3f}'.format(avg_theta_errors[0]))
