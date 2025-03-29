import cv2 as cv
import numpy as np

def colorize_image(input_path, output_path):
    # Load models and weights
    protoFile = "D:/Colorize BW Images/models/colorization_deploy_v2.prototxt"
    weightsFile = "D:/Colorize BW Images/models/colorization_release_v2.caffemodel"
    pts_in_hull = np.load("D:/Colorize BW Images/models/pts_in_hull.npy")

    # Load network
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

    # Load input image
    frame = cv.imread(input_path)
    img_rgb = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l = img_lab[:, :, 0]

    # Preprocess
    W_in, H_in = 224, 224
    img_l_rs = cv.resize(img_l, (W_in, H_in))
    img_l_rs -= 50

    net.setInput(cv.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_dec_us = cv.resize(ab_dec, (img_rgb.shape[1], img_rgb.shape[0]))

    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

    # Save colorized image to output_path
    cv.imwrite(output_path, (img_bgr_out * 255).astype(np.uint8))
    return output_path
