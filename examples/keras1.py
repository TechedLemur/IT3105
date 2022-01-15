# A few simple examples using Keras

from tensorflow import keras as KER
import doodler as DDL
import tflow.tflowtools as TFT
import numpy as np
import tensorflow as tf
import tensorflow.keras.models as KMOD
import tensorflow.keras.layers as KLAY
import tflow.kdtflowclasses as KDTFC
import tensorflow.keras.callbacks as KCALL
import datetime
import os


# **** EXAMPLE 1 ******

# Generate a simple net using "Sequential", which just adds each new layer in sequence, from
# input to output.
def gennet(num_classes=10, lrate=0.01, opt='SGD', loss='categorical_crossentropy', act='relu',
           conv=True, lastact='softmax'):
    opt = eval('KER.optimizers.' + opt)
    loss = eval('KER.losses.'+loss) if type(loss) == str else loss
    # The model can now be built sequentially from input to output
    model = KER.models.Sequential()
    if conv:  # Starting off with a convolution layer followed by max pooling
        model.add(KER.layers.Conv1D(16, kernel_size=(5)))
        model.add(KER.layers.MaxPooling1D(5))
        model.add(KER.layers.Flatten())
    model.add(KER.layers.Dense(50, activation=act))
    model.add(KER.layers.Dense(25, activation=act))
    model.add(KER.layers.Dense(num_classes, activation=lastact))
    model.compile(optimizer=opt(lr=lrate), loss=loss,
                  metrics=[KER.metrics.categorical_accuracy])
    return model

# Build the net more explicitly here (via KER.models.Model), giving more control.


def gennet2(num_classes=10, lrate=0.01, opt='SGD', loss='categorical_crossentropy', act='relu',
            conv=False, lastact='softmax', in_shape=(10,)):
    opt = eval('KER.optimizers.' + opt)
    loss = eval('KER.losses.'+loss)
    input = KER.layers.Input(shape=in_shape, name='input_layer')
    x = input
    if conv:  # Starting off with a convolution layer followed by max pooling
        # Each new layer creator takes upstream layer as input
        x = KER.layers.Conv1D(16, kernel_size=(5))(x)
        x = KER.layers.MaxPooling1D(5)(x)
        x = KER.layers.Flatten()(x)
    x = KER.layers.Dense(50, activation=act)(x)
    x = KER.layers.Dense(25, activation=act)(x)
    output = KER.layers.Dense(num_classes, activation=lastact)(x)
    # KERAS now knows how to connect input to output
    model = KER.models.Model(input, output)
    model.compile(optimizer=opt(lr=lrate), loss=loss,
                  metrics=[KER.metrics.categorical_accuracy])
    return model

#  The main routine for LEarning to COunt.  This counts segments in a vector of length = vlen
# Range of segment counts = (seg0, seg1), lrate = learning rate, vf = validation fraction


def leco(epochs=100, ncases=500, seg0=0, seg1=8, vlen=50, vf=0.2, lrate=0.1, act='relu',
         loss='MSE', conv=True):
    # *** Data Preparation ***
    tlen = seg1 - seg0 + 1  # length of target vectors
    cases = TFT.gen_segmented_vector_cases(
        vlen, ncases, minsegs=seg0, maxsegs=seg1, poptargs=True)
    in_shape = (vlen, 1) if conv else (vlen)  # 1 = no. input channels
    inputs = np.array([np.array(c[0]).reshape(in_shape)
                       for c in cases]).astype(np.float32)
    targets = np.array([c[1] for c in cases], dtype='float')
    # ** Build the neural net, and create a tensorboard callback
    nn = gennet(num_classes=tlen, lrate=lrate, act=act, conv=conv,
                loss=loss)  # Create the neural net (a.k.a. "model")
    tb_callback, logdir = KDTFC.gen_tensorboard_callback(nn)
    # This actually trains the model, with intermittent validation testing
    nn.fit(inputs, targets, epochs=epochs, batch_size=16, validation_split=vf, verbose=1,
           callbacks=[tb_callback])
    # To view, open browser to URL = localhost:6006
    KDTFC.fireup_tensorboard(logdir)
    return nn, logdir


# *********** EXAMPLE 2 ****************************

# Generate a convolution network.
def gencon(num_classes=10, lrate=0.01, opt='SGD', loss='categorical_crossentropy', act='relu'):
    opt = eval('KER.optimizers.' + opt)
    loss = eval('KER.losses.'+loss)
    # The model can now be built sequentially from input to output
    model = KER.models.Sequential()
    # The first layer can include the dims of the upstream layer (input_shape = in_dims)
    #   Otherwise, this part of the model will be configured during the call to model.fit().
    model.add(KER.layers.Conv2D(16, kernel_size=(
        3, 3), strides=(1, 1), activation=act))
    model.add(KER.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(KER.layers.Conv2D(32, kernel_size=(
        3, 3), activation=act, strides=(1, 1)))
    model.add(KER.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(KER.layers.Flatten())
    model.add(KER.layers.Dense(100, activation=act))
    model.add(KER.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=opt(lr=lrate), loss=loss,
                  metrics=[KER.metrics.categorical_accuracy])
    return model


# LEarn to COunt using the 2-d doodle diagrams.  Range of image counts = (im0, im1)

def leco2(epochs=100, im0=0, im1=8, ncases=500, dims=(50, 50), gap=1, vf=0.2, lrate=0.01, mbs=16, act='relu'):
    tlen = im1 - im0 + 1  # length of target vectors
    in_shape = list(dims) + [1]  # 1 = no. input channels

    # A simple, local function for generating 1-hot target vectors from an integer. E.g. 2 => (0 0 1 0 0 ...)
    def gentarget(k):
        targ = [0]*tlen
        targ[k] = 1
        return targ

    # Create a Doodler object for creating the doodle images
    d = DDL.Doodler(rows=dims[0], cols=dims[1], gap=gap, multi=True)
    cases = d.gen_random_cases(ncases, image_types=['ball', 'box', 'frame', 'ring'], flat=False,
                               wr=[0.1, 0.25], hr=[0.1, 0.25], figcount=(im0, im1))
    # Pull the inputs and targets out of the cases
    inputs = np.array([np.array(c[0]).reshape(
        in_shape).astype(np.float32) for c in cases])
    targets = np.array([gentarget(c[1]) for c in cases])
    # Generate the convolutional neural network
    # Create the neural net (a.k.a. "model")
    nn = gencon(num_classes=tlen, lrate=lrate, act=act)
    tb_callback, logdir = KDTFC.gen_tensorboard_callback(
        nn)  # Saving values for display in a tensorboard
    # Train with validation testing
    nn.fit(inputs, targets, epochs=epochs, batch_size=mbs,
           validation_split=vf, verbose=2, callbacks=[tb_callback])
    # Open a tensorboard in the browser (localhost:6006)
    KDTFC.fireup_tensorboard(logdir)
    return nn, logdir

# ******** EXAMPLE 3 ***********************
# Testing the use of SplitGD in KDTFC (kdtflowclasses.py) ****


def leco3(epochs=100, ncases=500, seg0=0, seg1=8, vlen=50, vf=0.2, lrate=0.001, act='relu', conv=True, mbs=16, verb=1, tb=True):
    # Data Preparation ****
    tlen = seg1 - seg0 + 1  # length of target vectors
    cases = TFT.gen_segmented_vector_cases(
        vlen, ncases, minsegs=seg0, maxsegs=seg1, poptargs=True)
    in_shape = (vlen, 1) if conv else (vlen)  # 1 = no. input channels
    inputs = np.array([np.array(c[0]).reshape(in_shape)
                       for c in cases]).astype(np.float32)
    targets = np.array([c[1] for c in cases])
    # *** Build the neural net (a.k.a. "model")
    nn = gennet2(num_classes=tlen, lrate=lrate,
                 act=act, conv=conv, in_shape=in_shape)
    # Training and Validation Testing
    sgd = KDTFC.SplitGD(nn)  # Make the split-gradient-descent object
    if tb:  # Using the TensorBoard, which is still not working properly with SplitGD.
        tb_callback, logdir = KDTFC.gen_tensorboard_callback(nn)
        cbs = [tb_callback]
    else:
        cbs, logdir = None, None
    sgd.fit(inputs, targets, epochs=epochs, mbs=mbs,
            vfrac=vf, verbosity=verb, callbacks=cbs)
    if tb:
        # Open a browser to URL = localhost:6006
        KDTFC.fireup_tensorboard(logdir)
    return nn, logdir

# ********* EXAMPLE 4 ****************************
# Testing the use of eligibility traces with a Keras model **************


def testelig(epochs=5, ncases=10, vlen=5, lrate=0.005, act='relu', lastact='sigmoid', opt='SGD'):
    in_layer = KLAY.Input(shape=vlen, name='nn_input')
    x = KLAY.Dense(12, activation=act)(in_layer)
    out_layer = KLAY.Dense(1, name='nn_output', activation=lastact)(x)
    # Create NN = "model"
    nn = KMOD.Model(in_layer, out_layer, name='elig_test_net')
    optimizer = eval('KOPT.' + opt)
    nn.compile(loss='MSE', optimizer=optimizer(learning_rate=lrate))
    nn.summary()
    tdelg = KDTFC.TDCriticEligTracer(nn)
    for _ in range(epochs):
        inputs = np.random.uniform(
            0, 1, size=(ncases, vlen)).astype(np.float32)
        targets = np.random.uniform(0, 1, size=(ncases, 1)).astype(np.float32)
        td_errors = np.random.uniform(-1, 1, size=len(inputs))
        tdelg.fit(inputs, targets, td_errors)
    return nn

# *** Auxiliary functions ***


def dumpit(nn, fname='junk'):
    KDTFC.save_keras_model(nn, fname=fname)


def loadit(fname='junk'):
    return KDTFC.load_keras_model(fname)

# ****************** TENSORBOARD SHORTCUT ******
# This creates a simple callback for using tensorboard to plot the error history


def gen_tensorboard_callback(model, basedir="tblog/fit/", freq=1, upfreq='epoch'):
    ldir = basedir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cb = KCALL.TensorBoard(log_dir=ldir, histogram_freq=freq,
                           update_freq=upfreq, write_graph=True)
    cb.set_model(model)
    return cb, ldir


def fireup_tensorboard(logdir='tblog/fit'):
    os.system('tensorboard --bind_all --logdir=' + logdir + ' --port 6006')

# Type in the URL localhost:6006, and the board should come up.


def clear_tensorflow_log(logdir):   os.system('rm -r ' + logdir + '*')
