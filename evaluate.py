import numpy as np
import os
from keras.backend import epsilon
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import cv2
from model import get_frontend, add_context, add_softmax
import argparse
import keras.backend as K

dir = '/home/nicholas/Documents/Other/benchmark_RELEASE/dataset/'

classes= ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
          'motorbike','person','potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
# 0=background
# 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
# 6=bus, 7=car, 8=cat, 9=chair, 10=cow
# 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

# Settings for the Pascal dataset
input_width, input_height = 900, 900
label_margin = 186
mean = [102.93, 111.36, 116.52]
has_context_module = True


def get_trained_model(args):
    """ Returns a model with loaded weights. """

    model = get_frontend(input_width, input_height)

    if has_context_module:
        model = add_context(model)

    model = add_softmax(model)

    def load_tf_weights():
        """ Load pretrained weights converted from Caffe to TF. """

        # 'latin1' enables loading .npy files created with python2
        weights_data = np.load(args.weights_path, encoding='latin1').item()

        for layer in model.layers:
            if layer.name in weights_data.keys():
                layer_weights = weights_data[layer.name]
                layer.set_weights((layer_weights['weights'],
                                   layer_weights['biases']))

    def load_keras_weights():
        """ Load a Keras checkpoint. """
        model.load_weights(args.weights_path)

    if args.weights_path.endswith('.npy'):
        load_tf_weights()
    elif args.weights_path.endswith('.hdf5'):
        load_keras_weights()
    else:
        raise Exception("Unknown weights format.")

    return model


def predict(val_img_names, val_mask_names):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', nargs='?',
                        default='/home/nicholas/Documents/Other/benchmark_RELEASE/dataset/img/2008_000033.jpg',
                        help='Required path to input image')
    parser.add_argument('--output_path', default='./images/2008_000033_seg.png',
                        help='Path to segmented image')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--zoom', default=8, type=int,
                        help='Upscaling factor')
    parser.add_argument('--weights_path', default='conversion/converted/dilation8_pascal_voc.npy',
                        help='Weights file')
    args = parser.parse_args()
    model = get_trained_model(args)

    pairs = list(zip(val_img_names, val_mask_names))
    y_pred_batch = []
    y_true_batch = []
    batch_size = pairs.__len__()
    for img, msk in pairs:
        # Load image and swap RGB -> BGR to match the trained weights
        image_rgb = np.array(Image.open(img)).astype(np.float32)
        image = image_rgb[:, :, ::-1] - args.mean
        image_size = image.shape

        # Network input shape (batch_size=1)
        net_in = np.zeros((1, input_height, input_width, 3), dtype=np.float32)

        output_height = input_height - 2 * label_margin
        output_width = input_width - 2 * label_margin

        # This simplified prediction code is correct only if the output
        # size is large enough to cover the input without tiling
        assert image_size[0] < output_height
        assert image_size[1] < output_width

        # Center pad the original image by label_margin.
        # This initial pad adds the context required for the prediction
        # according to the preprocessing during training.
        image = np.pad(image,
                       ((label_margin, label_margin),
                        (label_margin, label_margin),
                        (0, 0)), 'reflect')

        # Add the remaining margin to fill the network input width. This
        # time the image is aligned to the upper left corner though.
        margins_h = (0, input_height - image.shape[0])
        margins_w = (0, input_width - image.shape[1])
        image = np.pad(image,
                       (margins_h,
                        margins_w,
                        (0, 0)), 'reflect')

        # Run inference
        net_in[0] = image
        prob = model.predict(net_in)[0]

        # Reshape to 2d here since the networks outputs a flat array per channel
        prob_edge = np.sqrt(prob.shape[0]).astype(np.int)
        prob = prob.reshape((prob_edge, prob_edge, 21))
        h_crop = (image_size[0] // 8) + 1
        w_crop = (image_size[1] // 8) + 1
        prob = prob[:h_crop, :w_crop, :]
        prob = np.transpose(prob, (2, 0, 1))
        resize_prob = np.zeros((21, image_size[0], image_size[1]))

        for i in range(len(prob)):
            resize_prob[i] = cv2.resize(src=prob[i], dsize=(image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)

        prob = np.transpose(resize_prob, (1, 2, 0))


        # Recover the most likely prediction (actual segment class)
        prediction = np.argmax(prob, axis=2)
        mask = load_img(path=msk, grayscale=True)
        mask = img_to_array(mask)
        # prediction = prediction.flatten()
        # mask = mask.flatten()
        y_pred_batch.append(prediction)
        y_true_batch.append(mask)

    evaluate(y_pred_batch, y_true_batch)


def evaluate(y_pred_batch, y_true_batch):

    conf_m, IOU, meanIOU = calculate_iou(y_pred_batch=y_pred_batch, y_true_batch=y_true_batch)
    print(conf_m)
    print(IOU)
    print(meanIOU)


def calculate_iou(y_pred_batch, y_true_batch):
    conf_m = np.zeros((21, 21), dtype=float)
    for i in range(len(y_true_batch)):
        flat_pred = np.ravel(y_pred_batch[i]).astype(int)
        flat_label = np.ravel(y_true_batch[i]).astype(int)
        for p, l in zip(flat_pred, flat_label):
            if l == 0:
                continue
            if l < 21 and p < 21:
                conf_m[l, p] += 1
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU


def calculate_pixel_accuracy(y_pred_batch, y_true_batch, voc_class):
    nii = 0
    ti = 0
    for i in range(len(y_true_batch)):
        nii += np.sum(y_pred_batch == voc_class)
        ti += np.sum(y_true_batch == voc_class)
    return nii / (ti + epsilon())

def dice_loss(y_pred_batch, y_true_batch):
    smooth = 1.
    y_true_f = K.flatten(y_true_batch)
    y_pred_f = K.flatten(y_pred_batch)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def IoU(y_pred_batch, y_true_batch, voc_class):
    nii_sum = 0
    ti_sum = 0
    nji_sum = 0
    for i in range(len(y_true_batch)):
        y_true = np.reshape(y_true_batch[i], [y_true_batch[i].shape[0], y_true_batch[i].shape[1]])
        y_pred = y_pred_batch[i]
        y_true_class = y_true == voc_class
        y_pred_class = y_pred == voc_class
        nii = (y_true_class * y_pred_class)
        ti = y_true_class
        nji = 0

        for j in range(1, 21):
            ji = y_pred == j
            nji += np.sum(ji)

        nii_sum += np.sum(nii)
        ti_sum += np.sum(ti)
        nji_sum += np.sum(nji)

    return nii_sum / ((ti_sum + nji_sum + nii_sum) + epsilon())

# def pixelAccuracy(y_pred, y_true, voc_class):
#     y_true = np.reshape(y_true, [y_true.shape[0], y_true.shape[1]])
#     y_true_not_background = y_true > 0
#     y_true_class = y_true == voc_class
#     y_pred_class = y_pred == voc_class
#     ti = y_true == voc_class
#     nii = (y_true_class * y_pred_class)
#     nji = 0
#     for j in range(1, 21):
#         ji = y_pred == j
#         nji += np.sum(ji)
#
#     epsilon_val = epsilon()
#     return np.sum(nii) / ((np.sum(ti) + nji - np.sum(nii)) + epsilon_val)

if __name__ == "__main__":

    # Build absolute image paths
    def build_abs_paths(basenames):
        img_fnames = [os.path.join(dir + 'img', f) + '.jpg' for f in basenames]
        mask_fnames = [os.path.join(dir + 'pngs', f) + '.png' for f in basenames]
        return img_fnames, mask_fnames

    val_basenames = [l.strip() for l in open(dir+'val.txt').readlines()][:100]
    val_img_fnames, val_mask_fnames = build_abs_paths(val_basenames)
    predict(val_img_names=val_img_fnames, val_mask_names=val_mask_fnames)