import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import cv2
import random
import matplotlib.pyplot as plt



class TrafficSignClassifier(object):
    def __init__(self):
        self.define_class_names()
    
#===============================DATA/IMAGE OPERATIONS=====================================================
    
    def define_class_names(self):
        self.CLASSES = np.array([
            "Speed limit (20km/h)",
            "Speed limit (50km/h)",
            "Speed limit (30km/h)",
            "Speed limit (60km/h)",
            "Speed limit (70km/h)",
            "Speed limit (80km/h)",
            "End of speed limit (80km/h)",
            "Speed limit (100km/h)",
            "Speed limit (120km/h)",
            "No passing",
            "No passing for vehicles over 3.5 metric tons",
            "Right-of-way at the next intersection",
            "Priority road",
            "Yield",
            "Stop",
            "No vehicles",
            "Vehicles over 3.5 metric tons prohibited",
            "No entry",
            "General caution",
            "Dangerous curve to the left",
            "Dangerous curve to the right",
            "Double curve",
            "Bumpy road",
            "Slippery road",
            "Road narrows on the right",
            "Road work",
            "Traffic signals",
            "Pedestrians",
            "Children crossing",
            "Bicycles crossing",
            "Beware of ice/snow",
            "Wild animals crossing",
            "End of all speed and passing limits",
            "Turn right ahead",
            "Turn left ahead",
            "Ahead only",
            "Go straight or right",
            "Go straight or left",
            "Keep right",
            "Keep left",
            "Roundabout mandatory",
            "End of no passing",
            "End of no passing by vehicles over 3.5 metric tons",
        ])
        
    def read_data(self):
        training_file = './traffic-signs-data/train.p'
        validation_file= './traffic-signs-data/valid.p'
        testing_file = './traffic-signs-data/test.p'
        
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        
        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']

        n_train = self.X_train.shape[0]
        n_validation = self.X_valid.shape[0]
        n_test = self.X_test.shape[0]
        
        image_shape = self.X_train.shape[1:3]
        
        df = pd.read_csv('signnames.csv')
        self.n_classes = len(df.index)
        
        print("Number of training examples =", n_train)
        print("Number of testing examples =", n_test)
        print("Image data shape =", image_shape)
        print("Number of classes =", self.n_classes)

    def show_im(self, image):
        plt.figure(figsize=(1,1))
        plt.imshow(image, cmap="gray")
        plt.show()

    def show_im_cv2(self, image):
        cv2.imshow("Window", image)
        cv2.waitKey(0)

    def show_images_of_class(self, class_num, num_images=5):
        indices = np.argwhere(self.y_train == class_num)
        for index in indices[:num_images]:
            image = self.X_train[index]
            image = (image + 1) * 128
            self.show_im(image.squeeze().astype(np.int32))

    def normalize(self):
        # print(np.mean(self.X_train[0]))
        self.X_train = (self.X_train/128)-1
        self.X_valid = (self.X_valid/128)-1
        self.X_test = (self.X_test/128)-1
        # print(np.mean(self.X_train[0]))

    def prepare_my_test_images(self):
        def add_image_and_class(image, class_num):
            # image = np.expand_dims(image, 0)
            image = (image/128)-1
            # self.show_im_cv2(image)
            image = image[:, :, ::-1]
            # self.show_im_cv2(image)
            # self.show_im(image)
            # class_num = np.array([class_num], dtype=np.int32)
            self.my_images.append(image)
            self.my_classes.append(int(class_num))

        self.my_images = []
        self.my_classes = []

        add_image_and_class(cv2.imread('./test_images/70km_h.png'), 4)
        add_image_and_class(cv2.imread('./test_images/stop_sign.png'), 14)
        add_image_and_class(cv2.imread('./test_images/yield_sign.png'), 13)
        add_image_and_class(cv2.imread('./test_images/keep_left_sign.png'), 39)
        add_image_and_class(cv2.imread('./test_images/snow_sign.png'), 30)
        add_image_and_class(cv2.imread('./test_images/roundabout_sign.png'), 40)
        add_image_and_class(cv2.imread('./test_images/stop_sign_2.png'), 14)
        # add_image_and_class(cv2.imread('./test_images/slippery_road_sign.png'), 23) # sign not included in training set
        add_image_and_class(cv2.imread('./test_images/slippery_road_sign_2.png'), 23)
        add_image_and_class(cv2.imread('./test_images/priority_road_sign.png'), 12)
        add_image_and_class(cv2.imread('./test_images/road_work_sign.png'), 25)

    #=========================TF OPERATIONS=========================================================

    def LeNet(self, x):    
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1
        
        # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        self.conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
        # SOLUTION: Activation.
        self.conv1_r = tf.nn.relu(self.conv1)
    
        # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
        self.conv1_maxp = tf.nn.max_pool(self.conv1_r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
        # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        self.conv2 = tf.nn.conv2d(self.conv1_maxp, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        
        # SOLUTION: Activation.
        conv2_r = tf.nn.relu(self.conv2)
    
        # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2_maxp = tf.nn.max_pool(conv2_r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
        # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
        fc0   = flatten(conv2_maxp)
        
        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
        
        # SOLUTION: Activation.
        fc1    = tf.nn.relu(fc1)
    
        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(84))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
        
        # SOLUTION: Activation.
        fc2    = tf.nn.relu(fc2)
    
        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, self.n_classes), mean = mu, stddev = sigma))
        fc3_b  = tf.Variable(tf.zeros(self.n_classes))
        logits = tf.matmul(fc2, fc3_W) + fc3_b
        
        return logits

    def set_hyperparameters(self):
        self.rate = 0.001
        self.epochs = 50
        self.batch_size = 128
    
    def define_tf_graph(self):
        if not hasattr(self, 'rate'):
            self.set_hyperparameters()

        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.one_hot_y = tf.one_hot(self.y, self.n_classes)

        
        self.logits = self.LeNet(self.x)
        self.top_five_probs = tf.nn.top_k(self.logits, k=5)
        self.softmax = tf.nn.softmax(self.logits, axis=1)
        self.softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        
        loss_operation = tf.reduce_mean(self.softmax_cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.rate)
        self.training_operation = optimizer.minimize(loss_operation)
        
        
        correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.one_hot_y, axis=1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.output_prediction = tf.argmax(self.logits, axis=1)

        self.saver = tf.train.Saver()
        self.save_loc = './lenet.ckpt'

    def evaluate(self, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self.batch_size):
            batch_x, batch_y = X_data[offset:offset+self.batch_size], y_data[offset:offset+self.batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(self.X_train)
    
            print("Training...")
            print()
            for i in range(self.epochs):
                self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
                for offset in range(0, num_examples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x, batch_y = self.X_train[offset:end], self.y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y})
    
                validation_accuracy = self.evaluate(self.X_valid, self.y_valid)
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()
    
            self.saver.save(sess, self.save_loc)
            print("Model saved")
    
    def test(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, self.save_loc)
    
            test_accuracy = self.evaluate(self.X_test, self.y_test)
    
        print('Test accuracy on test set: {}'.format(test_accuracy))

    def test_on_my_images(self):
        if not hasattr(self, 'my_images'):
            self.prepare_my_test_images()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, self.save_loc)

            print("Testing on my images...")
            num_correct = 0

            # one_hot_y_out = sess.run(self.one_hot_y, feed_dict={self.x: self.my_images, self.y: self.my_classes})
            # logits_out = sess.run(self.logits, feed_dict={self.x: self.my_images, self.y: self.my_classes})
            softmax_values = sess.run(self.softmax, feed_dict={self.x: self.my_images, self.y: self.my_classes})
            # sm_c_entropy = sess.run(self.softmax_cross_entropy, feed_dict={self.x: self.my_images, self.y: self.my_classes})
            output = sess.run(self.output_prediction, feed_dict={self.x: self.my_images, self.y: self.my_classes})

            top5 = sess.run(self.top_five_probs, feed_dict={self.x: self.my_images, self.y: self.my_classes})

            print()
            np.set_printoptions(suppress=True, precision=3)
            for i, (output_class, real_class) in enumerate(zip(output, self.my_classes)):
                num_correct = num_correct + 1 if output_class == real_class else num_correct
                print("The Real Answer:", self.CLASSES[real_class])
                print("Network's Prediction, in order of confidence:", self.CLASSES[top5.indices[i]])
                print("Network Softmaxes (%)", softmax_values[i, top5.indices[i]] * 100)
                print("Logits Top 5 values: ", top5.values[i])
                print()

            print("Num of My {} Images That it Classified Correctly: {}".format(len(self.my_images), num_correct))

    def outputFeatureMap(self, image_input, tf_activations, activation_min=-1, activation_max=-1, plt_num=1):

        image_input = (image_input / 128) - 1
        image_input = image_input[:, :, ::-1]

        # plt.figure(plt_num, figsize=(15,15))
        tf_activations = np.array(tf_activations)
        rows = tf_activations.shape[0]
        cols = 0
        for tf_act in tf_activations:
            cols = tf_act.shape[3] if tf_act.shape[3] > cols else cols
        fig, axes = plt.subplots(nrows=int(rows), ncols=int(cols))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i, tf_activation in enumerate(tf_activations):
                activation = tf_activation.eval(session=sess, feed_dict={self.x: image_input})
                # print("Activation Shape:", activation.shape)
                featuremaps = activation.shape[3]
                for featuremap in range(featuremaps):
                    # axes[i, featuremap] = plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
                    # plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
                    if activation_min != -1 & activation_max != -1:
                        # plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
                        axes[i, featuremap].imshow(activation[0, :, :, featuremap], interpolation="nearest",
                                                   vmin=activation_min, vmax=activation_max, cmap="gray")
                    elif activation_max != -1:
                        # plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
                        axes[i, featuremap].imshow(activation[0, :, :, featuremap], interpolation="nearest",
                                                   vmax=activation_max, cmap="gray")
                    elif activation_min != -1:
                        # plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
                        axes[i, featuremap].imshow(activation[0, :, :, featuremap], interpolation="nearest",
                                                   vmin=activation_min, cmap="gray")
                    else:
                        # plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
                        axes[i, featuremap].imshow(activation[0, :, :, featuremap], interpolation="nearest",
                                                   cmap="gray")
        plt.show()

            
#=====================================================================================================
        
    def run_pipeline(self):
        self.read_data()
        self.normalize()
        # self.show_images_of_class(class_num=25, num_images=7)
        self.define_tf_graph()
        self.test()
        self.prepare_my_test_images()
        self.test_on_my_images()

        test_im = np.expand_dims(cv2.imread('./test_images/road_work_sign.png'), 0)
        self.outputFeatureMap(test_im, [self.conv1, self.conv1_r, self.conv1_maxp])

classifier = TrafficSignClassifier()
classifier.run_pipeline()



