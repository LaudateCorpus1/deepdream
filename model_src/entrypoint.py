import io
import os.path
import sys

import PIL.Image
from PIL import Image

import caffe
import numpy as np
import scipy.ndimage as nd
from google.protobuf import text_format

import ai_integration


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    # print np.float32(img).shape
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


def objective_L2(dst):
    dst.diff[:] = dst.data


# objective for guided dreaming
def objective_guide(dst, guide_features):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch, -1)
    y = y.reshape(ch, -1)
    A = x.T.dot(y)  # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch, -1)[:] = y[:, A.argmax(1)]  # select ones that match best


# from https://github.com/jrosebr1/bat-country/blob/master/batcountry/batcountry.py
def prepare_guide(net, image, end="inception_4c/output", maxW=224, maxH=224):
    # grab dimensions of input image
    (w, h) = image.size

    # GoogLeNet was trained on images with maximum width and heights
    # of 224 pixels -- if either dimension is larger than 224 pixels,
    # then we'll need to do some resizing
    if h > maxH or w > maxW:
        # resize based on width
        if w > h:
            r = maxW / float(w)

        # resize based on height
        else:
            r = maxH / float(h)

        # resize the image
        (nW, nH) = (int(r * w), int(r * h))
        image = np.float32(image.resize((nW, nH), PIL.Image.BILINEAR))

    (src, dst) = (net.blobs["data"], net.blobs[end])
    src.reshape(1, 3, nH, nW)
    src.data[0] = preprocess(net, image)
    net.forward(end=end)
    guide_features = dst.data[0].copy()

    return guide_features


# -------
# Make dreams
# -------
def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data']  # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter shift

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size / np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255 - bias)


def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True,
              **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src.reshape(1, 3, h, w)  # resize the network's input image size
        src.data[0] = octave_base + detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip:  # adjust image contrast if clipping is disabled
                vis = vis * (255.0 / np.percentile(vis, 99.98))

        # extract details produced on the current octave
        detail = src.data[0] - octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


# --------------
# Guided Dreaming
# --------------
def make_step_guided(net, step_size=1.5, end='inception_4c/output',
                     jitter=32, clip=True, objective_fn=objective_guide, **objective_params):
    '''Basic gradient ascent step.'''

    # if objective_fn is None:
    #    objective_fn = objective_L2

    src = net.blobs['data']  # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter shift

    net.forward(end=end)
    objective_fn(dst, **objective_params)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size / np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255 - bias)


def deepdream_guided(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output',
                     clip=True, objective_fn=objective_guide, **step_params):
    # if objective_fn is None:
    #    objective_fn = objective_L2

    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src.reshape(1, 3, h, w)  # resize the network's input image size
        src.data[0] = octave_base + detail
        for i in xrange(iter_n):
            make_step_guided(net, end=end, clip=clip, objective_fn=objective_fn, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip:  # adjust image contrast if clipping is disabled
                vis = vis * (255.0 / np.percentile(vis, 99.98))

        # extract details produced on the current octave
        detail = src.data[0] - octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


layersloop = ['inception_4c/output', 'inception_4d/output',
              'inception_4e/output', 'inception_5a/output',
              'inception_5b/output', 'inception_5a/output',
              'inception_4e/output', 'inception_4d/output',
              'inception_4c/output']

model_path = '../models/bvlc_googlenet/'
model_name = 'bvlc_googlenet.caffemodel'

if not os.path.exists(model_path):
    print("Model directory not found")
    print("Please set the model_path to a correct caffe model directory")
    sys.exit(0)

model = os.path.join(model_path, model_name)

if not os.path.exists(model):
    print("Model not found")
    print("Please set the model_name to a correct caffe model")
    print("or download one with ./caffe_dir/scripts/download_model_binary.py caffe_dir/models/bvlc_googlenet")
    sys.exit(0)

gpu = 0
guide_image = None
octaves = 4
octave_scale = 1.5
iterations = 5
jitter = 32
stepsize = 1.5

# Load DNN
net_fn = model_path + 'deploy.prototxt'
param_fn = model_path + model_name  # 'bvlc_googlenet.caffemodel'

if gpu is None:
    print("SHITTTTTTTTTTTTTT You're running CPU man =D")
else:
    caffe.set_mode_gpu()
    caffe.set_device(int(gpu))
    print("GPU mode [device id: %s]" % gpu)
    print("using GPU, but you'd still better make a cup of coffee")

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean=np.float32([104.0, 116.0, 122.0]),  # ImageNet mean, training set dependent
                       channel_swap=(2, 1, 0))  # the reference model has channels in BGR order instead of RGB

while True:
    with ai_integration.get_next_input(inputs_schema={
        "image": {
            "type": "image"
        }
    }) as inputs_dict:
        frame = Image.open(io.BytesIO(inputs_dict['image']))

        frame = np.float32(frame)

        # TODO THIS INDEX Should be fiddled until it looks good
        endparam = layersloop[0]

        # Choosing between normal dreaming, and guided dreaming
        if guide_image is None:
            frame = deepdream(net, frame, iter_n=iterations,
                              step_size=stepsize, octave_n=octaves, octave_scale=octave_scale, jitter=jitter,
                              end=endparam)
        else:
            print('Setting up Guide with selected image')
            guide_features = prepare_guide(net, PIL.Image.open(guide_image), end=endparam)

            frame = deepdream_guided(net, frame, iter_n=iterations,
                                     step_size=stepsize, octave_n=octaves, octave_scale=octave_scale,
                                     jitter=jitter, end=endparam, objective_fn=objective_guide,
                                     guide_features=guide_features, )

        imgByteArr = io.BytesIO()
        PIL.Image.fromarray(np.uint8(frame)).save(imgByteArr, format='JPEG', subsampling=0, quality=98)
        imgByteArr = imgByteArr.getvalue()

        result = {
            'content-type': 'image/jpeg',
            'data': imgByteArr,
            'success': True,
            'error': None
        }
        ai_integration.send_result(result)
