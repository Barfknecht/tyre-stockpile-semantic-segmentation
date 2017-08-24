import keras
import numpy as np
import sklearn.metrics as sklm
from PIL import Image
import cv2


input_width, input_height = 900, 900
label_margin = 186

mean=[102.93, 111.36, 116.52]

class Metrics(keras.callbacks.Callback):

    def __init__(self, val_img_names, val_mask_names):
        self.val_img_names = val_img_names
        self.val_mask_names = val_mask_names

    def on_epoch_end(self, epoch, logs={}):


        pairs = list(zip(self.val_img_names, self.val_mask_names))

        for img, msk in pairs:
            # Load image and swap RGB -> BGR to match the trained weights
            image_rgb = np.array(Image.open(img)).astype(np.float32)
            image = image_rgb[:, :, ::-1] - mean
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
            prob = self.model.predict(net_in)[0]

            # Reshape to 2d here since the networks outputs a flat array per channel
            prob_edge = np.sqrt(prob.shape[0]).astype(np.int)
            prob = prob.reshape((prob_edge, prob_edge, 21))
            prob_shape = prob.shape
            prob = np.transpose(prob, (2, 0, 1))
            resize_prob = np.zeros((prob_shape[2], image_size[0], image_size[1]))
            for i in range(len(prob)):
                resize_prob[i] = cv2.resize(src=prob[i], dsize=(image_size[1], image_size[0]))

            # Recover the most likely prediction (actual segment class)
            prob = np.transpose(resize_prob, (1, 2, 0))
            prediction = np.argmax(prob, axis=2)
            flatten_mask = np.array(msk).flatten()
            flatten_prediction = prediction.flatten()



        # score = np.asarray(self.model.predict(self.validation_data[0]))
        # predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        # targ = self.validation_data[1]
        #
        # self.auc.append(sklm.roc_auc_score(targ, score))
        # self.confusion.append(sklm.confusion_matrix(targ, predict))
        # self.precision.append(sklm.precision_score(targ, predict))
        # self.recall.append(sklm.recall_score(targ, predict))
        # self.f1s.append(sklm.f1_score(targ, predict))
        # self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        return