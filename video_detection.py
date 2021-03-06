import cv2
import tensorflow as tf
import numpy as np
import os
import glob
import argparse
import networks
import shutil


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="Path to video file.", type=str)
    parser.add_argument("--video_target_path", help="Path to output video file.", type=str)
    parser.add_argument("--model_path", help="Path to the pretrained classifier model", type=str)
    args = parser.parse_args()

    return args.video_path, args.video_target_path, args.model_path


def resize_image(image, target_shape):
    h, w = image.shape[0], image.shape[1]

    if h > w:
        pad = int(abs(h - w) / 2)
        if (h - w) % 2 == 0:
            image = np.pad(image, ((0, 0), (pad, pad)))
        else:
            image = np.pad(image, ((0, 0), (pad + 1, pad)))

    elif h < w:
        pad = int(abs(h - w) / 2)
        if (h - w) % 2 == 0:
            image = np.pad(image, ((pad, pad), (0, 0)))
        else:
            image = np.pad(image, ((pad + 1, pad), (0, 0)))

    image = cv2.resize(image, dsize=(target_shape, target_shape), interpolation=cv2.INTER_LINEAR)

    return image


def detect_objects(video_path, video_target_path, classifier_path):

    classification_threshold = 0.55
    id_to_class = {
        0: "T - shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    with tf.Session() as session:

        x_in = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="input")
        with tf.variable_scope("prediction"):
            logits = networks.wide_resnet(x_in, is_training=True)
        prob = tf.nn.softmax(logits)

        saver = tf.train.Saver()
        saver.restore(session, classifier_path)

        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(video_target_path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (contours, _) = cv2.findContours(image=image_gray,
                                             mode=cv2.RETR_EXTERNAL,
                                             method=cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            bounding_boxes = []
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                if min(w, h) >= 4:
                    bounding_boxes.append((x, y, w, h))
                    crop_img = image_gray[y:y + h, x:x + w]
                    resized_image = resize_image(crop_img, 28)
                    regions.append(resized_image)

            regions = np.reshape(np.array(regions), newshape=(-1, 28, 28, 1)).astype(float)
            prob_np = session.run(prob, feed_dict={x_in: regions})

            classes = np.argmax(prob_np, axis=1)
            classes = np.where(np.max(prob_np, axis=1) >= classification_threshold, classes,
                               np.ones(shape=classes.shape) * -1)

            for class_prob, class_id, bb in zip(prob_np, classes, bounding_boxes):

                if class_id != -1:
                    txt = id_to_class[class_id] + ' ' + "{:.0f}%".format(np.max(class_prob)*100)
                    cv2.putText(frame, txt, (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                                (255, 255, 255))
                    cv2.rectangle(img=frame,
                                  pt1=(bb[0], bb[1]),
                                  pt2=(bb[0] + bb[2], bb[1] + bb[3]),
                                  color=(255, 255, 255),
                                  thickness=1)

            out.write(frame)

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    video_path, video_target_path, classifier_path = parse_arguments()
    detect_objects(video_path, video_target_path, classifier_path)