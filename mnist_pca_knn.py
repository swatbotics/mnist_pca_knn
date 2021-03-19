########################################################################
#
# File:   mnist_pca_knn.py
# Author: Matt Zucker
# Date:   March 2021
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# Shows how to do kNN classification (plus optional PCA) on MNIST
# dataset.

import os, sys, struct, shutil, urllib.request, zipfile
import datetime
import cv2
import numpy as np

MNIST_IMAGE_SIZE = 28
MNIST_DIMS = 784

TARGET_DISPLAY_SIZE = 340

WINDOW_NAME = 'MNIST PCA + kNN Demo'

BATCH_SIZE = 500


######################################################################
# Read a single 4-byte integer from a data file

def read_int(f):
    buf = f.read(4)
    data = struct.unpack('>i', buf)
    return data[0]

######################################################################
# Load MNIST data from original file format

def parse_mnist(labels_file, images_file):

    labels = open(labels_file, 'rb')
    images = open(images_file, 'rb')

    lmagic = read_int(labels)
    assert lmagic == 2049

    lcount = read_int(labels)

    imagic = read_int(images)
    assert imagic == 2051

    icount = read_int(images)
    rows = read_int(images)
    cols = read_int(images)

    assert rows == cols
    assert rows == MNIST_IMAGE_SIZE

    assert icount == lcount 

    l = np.fromfile(labels, dtype='uint8')
    i = np.fromfile(images, dtype='uint8')

    i = i.reshape((icount,rows,cols))

    return l, i

######################################################################
# Download and parse MNIST data

def get_mnist_data():

    filenames = [
        'train-labels-idx1-ubyte', 
        'train-images-idx3-ubyte', 
        't10k-images-idx3-ubyte', 
        't10k-labels-idx1-ubyte'
    ]

    if not all([os.path.exists(name) for name in filenames]):

        downloaded = False

        if not os.path.exists('mnist.zip'):
            print('downloading mnist.zip...')
            url = 'http://mzucker.github.io/swarthmore/mnist.zip'
            req = urllib.request.Request(url)
            f = urllib.request.urlopen(req)
            with open('mnist.zip', 'wb') as ostr:
                shutil.copyfileobj(f, ostr)
            print('done\n')
            downloaded = True

        z = zipfile.ZipFile('mnist.zip', 'r')

        names = z.namelist()

        assert set(names) == set(filenames)

        print('extracting mnist.zip...')

        for name in names:
            print(' ', name)
            with z.open(name) as f:
                with open(name, 'wb') as ostr:
                    shutil.copyfileobj(f, ostr)

        if downloaded:
            os.unlink('mnist.zip')

        print('done\n')

    print('loading MNIST data...')

    train_labels, train_images = parse_mnist('train-labels-idx1-ubyte',
                                             'train-images-idx3-ubyte')

    test_labels, test_images = parse_mnist('t10k-labels-idx1-ubyte',
                                           't10k-images-idx3-ubyte')

    print('done\n')

    return train_labels, train_images, test_labels, test_images

######################################################################
# For majority voting step of k nearest neighbors
# https://stackoverflow.com/questions/19201972/can-numpy-bincount-work-with-2d-arrays

def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)
    return count

######################################################################
# Construct an object we can use for fast nearest neighbor queries.
# See https://github.com/mariusmuja/flann
# And https://docs.opencv.org/master/dc/de2/classcv_1_1FlannBasedMatcher.html

def get_knn_matcher():

    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    return matcher

######################################################################
# Use the matcher object to match query vectors to training vectors.
#
# Parameters:
#
#   * query_vecs is p-by-n
#   * train_vecs is m-by-n
#   * train_labels is flat array of length m (optional)
#
# Returns: 
#
#   * match_indices p-by-k indices of closest rows in train_vecs
#   * labels_pred flat array of length p (if train_labels is provided)

def match_knn(matcher, k, query_vecs, train_vecs, train_labels=None):

    knn_result = matcher.knnMatch(query_vecs, train_vecs, k)

    match_indices = np.full((len(query_vecs), k), -1, int)

    for i, item_matches in enumerate(knn_result):
        match_indices[i] = [ match.trainIdx for match in item_matches ]

    if train_labels is None:
        return match_indices

    match_labels = train_labels[match_indices]

    bcount = bincount2d(match_labels, bins=10)

    labels_pred = bcount.argmax(axis=1)

    return match_indices, labels_pred

######################################################################
# Draw outlined text in an image

def outline_text(img, text, pos, scl, fgcolor=None, bgcolor=None):

    if fgcolor is None:
        fgcolor = (255, 255, 255)

    if bgcolor is None:
        bgcolor = (0, 0, 0)

    for (c, w) in [(bgcolor, 3), (fgcolor, 1)]:

        cv2.putText(img, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scl, c, w, cv2.LINE_AA)

######################################################################
# Make a text image

def show_text_screen(text):

    menu = np.full((TARGET_DISPLAY_SIZE, TARGET_DISPLAY_SIZE, 3),
                   255, np.uint8)

    x = 10
    y = 20
    l = 20
    sz = 0.5
    
    for line in text:

        if line:

            cv2.putText(menu, line, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        sz, (0, 0, 0), 1, cv2.LINE_AA)

        y += l

    cv2.imshow(WINDOW_NAME, menu)

    while True:
        k = cv2.waitKey(5)
        if k >= 0:
            break

    return k

######################################################################
# Convert a row vector to an image.

def row2img(row, sz, text, img_type='image', bgcolor=None):

    scl = sz // MNIST_IMAGE_SIZE

    if img_type == 'eigenvector':
        rmax = np.abs(row).max()
        row = 0.5 + 0.5*row/rmax
    elif img_type == 'error':
        row = np.clip(0.5 + 0.5*row/255, 0, 1)
    else:
        row = np.clip(row/255, 0, 1)

    row = 1 - row

    img = (row * 255).astype(np.uint8).reshape(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)

    img = cv2.resize(img, (0, 0), fx=scl, fy=scl, interpolation=cv2.INTER_NEAREST)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if text is not None:
        outline_text(img, text, (4, 16), 0.5, bgcolor=bgcolor)

    return img
    
######################################################################
# Do classification demo

def demo_classification(train_images, train_vecs, train_labels,
                        test_images, test_vecs, test_labels,
                        matcher, knn_k):

    instructions = [
        'kNN Classification',
        '',
        'Hit [ or ] for prev/next image',
        'Hit { or } for prev/next error',
        'Hit R to jump to random image',
        'Hit ESC when done',
        '',
        'Hit any key to begin',
    ]

    show_text_screen(instructions)

    test_idx = -1
    delta = 1
    scroll_till_error = False

    num_test = len(test_images)

    while True:

        while True:

            test_idx = (test_idx + delta) % num_test
        
            query_vecs = test_vecs[test_idx:test_idx+1]
            query_label = test_labels[test_idx]
            query_image = test_images[test_idx]

            idx, label_pred = match_knn(matcher, knn_k, query_vecs,
                                        train_vecs, train_labels)

            idx = idx.flatten()
            label_pred = label_pred.squeeze()

            is_correct = (label_pred == query_label)

            if not is_correct or not scroll_till_error:
                break

        test_big = row2img(query_image, TARGET_DISPLAY_SIZE,
                           f'test image {test_idx:04d} label={query_label}')

        match_small = []

        smol_sz = TARGET_DISPLAY_SIZE // knn_k

        for i in idx:
            if knn_k <= 3:
                label = f'label={train_labels[i]}'
            elif knn_k <= 7:
                label = str(train_labels[i])
            else:
                label = None
            img = row2img(train_images[i], smol_sz, label,
                          bgcolor=(127, 127, 127))
                          
            match_small.append(img)

        h = test_big.shape[0]
        plen = h - img.shape[0]*len(idx)
        assert plen >= 0

        if plen:
            padding = np.full((plen, img.shape[1], 3), 255, np.uint8)
            match_small.append(padding)

        matches = np.vstack(tuple(match_small))

        display = np.hstack((test_big, matches))

        if is_correct:
            text = 'correct'
            color = (255, 0, 0)
        else:
            text = 'INCORRECT'
            color = (0, 0, 255)

        text = f'prediction={label_pred}, result={text}'

        outline_text(display, text, (4, h-8), 0.5, bgcolor=color)
    
        cv2.imshow(WINDOW_NAME, display)

        while True:
            k = cv2.waitKey(5)
            if k == 27:
                return
            elif k == ord(' ') or k == ord(']'):
                delta = 1
                scroll_till_error = False
                break
            elif k == ord('}'): 
                delta = 1
                scroll_till_error = True
                break
            elif k == ord('['):
                delta = num_test - 1
                scroll_till_error = False
                break
            elif k == ord('{'): 
                delta = num_test - 1
                scroll_till_error = True
                break
            elif k == ord('r') or k == ord('R'):
                delta = np.random.randint(num_test)
                scroll_till_error = False
                break

######################################################################
# Do PCA demo

def demo_mean_eigenvectors(mean, eigenvectors):

    instructions = [
        'Mean and eigenvectors',
        '',
        'Hit [ or ] for prev/next image',
        'Hit ESC when done',
        '',
        'Hit any key to begin',
    ]

    show_text_screen(instructions)

    images = [ row2img(mean.flatten(),
                       TARGET_DISPLAY_SIZE, 'Mean digit image') ]

    for i, evec in enumerate(eigenvectors):

        label = f'Eigenvector {i+1}/{len(eigenvectors)}'
        images.append(row2img(evec, TARGET_DISPLAY_SIZE, label, 
                              img_type='eigenvector'))
                              

    idx = 0
    n = len(images)

    while True:

        cv2.imshow(WINDOW_NAME, images[idx])
        
        k = cv2.waitKey(5)

        if k == ord(' ') or k == ord(']'):
            idx = (idx + 1) % n
        elif k == ord('['):
            idx = (idx + n - 1) % n
        elif k == 27:
            return

######################################################################
# Do reconstruction demo

def demo_reconstruction(mean, eigenvectors, test_images, test_vecs):

    instructions = [
        'Reconstruction from PCA',
        '',
        'Hit [ or ] for prev/next image',
        'Hit R for random image',
        'Hit ESC when done',
        '',
        'Hit any key to begin',
    ]

    show_text_screen(instructions)

    test_recons = cv2.PCABackProject(test_vecs, mean, eigenvectors)

    idx = 0
    n = len(test_images)
    
    while True:

        img_orig = test_images[idx]
        img_recons = test_recons[idx]

        orig = row2img(img_orig, TARGET_DISPLAY_SIZE, 
                       f'Test image {idx:04d}')

        recons = row2img(img_recons, TARGET_DISPLAY_SIZE,
                         f'Reconstructed from {len(eigenvectors)} PCs')

        err = row2img(img_orig - img_recons, TARGET_DISPLAY_SIZE,
                      'Error image', img_type='error')

        display = np.hstack((orig, recons, err))

        cv2.imshow(WINDOW_NAME, display)

        k = cv2.waitKey(5)

        if k == ord(' ') or k == ord(']'):
            idx = (idx + 1) % n
        elif k == ord('['):
            idx = (idx + n - 1) % n
        elif k == ord('r') or k == ord('R'):
            idx = np.random.randint(n)
        elif k == 27:
            return

######################################################################
# Run overall interactive demo

def interactive_demo(mean, eigenvectors, 
                     train_images, train_vecs, train_labels,
                     test_images, test_vecs, test_labels,
                     matcher, knn_k):


    wflags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(WINDOW_NAME, wflags)
    cv2.moveWindow(WINDOW_NAME, 50, 50)

    if mean is None or eigenvectors is None:
        
        demo_classification(train_images, train_vecs, train_labels,
                            test_images, test_vecs, test_labels,
                            matcher, knn_k)

        return

    text = [
        'Hit a key:',
        '',
        '1 - Show mean and eigenvectors',
        '2 - Reconstruction demo',
        '3 - Classification demo',
        '',
        'ESC - Quit'
    ]

    while True:

        k = show_text_screen(text)

        if k == 27:
            return
        elif k == ord('1'):
            demo_mean_eigenvectors(mean, eigenvectors)
        elif k == ord('2'):
            demo_reconstruction(mean, eigenvectors, test_images, test_vecs)
        elif k == ord('3'):
            demo_classification(train_images, train_vecs, train_labels,
                                test_images, test_vecs, test_labels,
                                matcher, knn_k)

######################################################################
# Load precomputed PCA mean and eigenvectors from an .npz file 
# (or create the file if it doesn't exist)

def load_precomputed_pca(train_images, k):

    try:

        d = np.load('mnist_pca.npz')
        mean = d['mean']
        eigenvectors = d['eigenvectors']
        print('loaded precomputed PCA from mnist_pca.npz')

    except:

        print('precomputing PCA one time only for train_images...')
        
        ndim = train_images.shape[1]
        mean, eigenvectors = cv2.PCACompute(train_images, 
                                            mean=None, 
                                            maxComponents=train_images.shape[1])

        print('done\n')

        np.savez_compressed('mnist_pca.npz',
                            mean=mean,
                            eigenvectors=eigenvectors)


    eigenvectors = eigenvectors[:k]

    return mean, eigenvectors

######################################################################

def main():

    if len(sys.argv) not in (3, 4):
        print('usage: python mnist_pca_knn.py [-i] PCA_K KNN_K')        
        print()
        print('note: set PCA_K to 0 to disable PCA')
        print()
        sys.exit(1)

    args = sys.argv[1:]
    
    try:
        interactive_idx = args.index('-i')
        args.pop(interactive_idx)
        interactive = True
    except:
        interactive = False

    assert len(args) == 2

    pca_k = int(args[0])
    assert pca_k >= 0 and pca_k <= MNIST_DIMS

    knn_k = int(args[1])
    assert knn_k > 0 and knn_k < 12

    train_labels, train_images, test_labels, test_images = get_mnist_data()

    train_images = train_images.astype(np.float32)
    train_images = train_images.reshape(-1, MNIST_DIMS) # make row vectors 

    if pca_k > 0:

        # note we could use cv2.PCACompute to do this but
        # instead we use a pre-computed eigen-decomposition
        # of the data if available
        mean, eigenvectors = load_precomputed_pca(train_images, pca_k)

        print('reducing dimensionality of training set...')

        train_vecs = cv2.PCAProject(train_images, mean, eigenvectors)

        print('done\n')

    else:

        mean = None
        eigenvectors = None
        train_vecs = train_images

    print('train_images:', train_images.shape)
    print('train_vecs:', train_vecs.shape)
    print()

    test_images = test_images.astype(np.float32)
    test_images = test_images.reshape(-1, MNIST_DIMS)

    if pca_k > 0:

        print('reducing dimensionality of test set...')
        test_vecs = cv2.PCAProject(test_images, mean, eigenvectors)

        print('done\n')

    else:

        test_vecs = test_images

    print('test_images:', test_images.shape)
    print('test_vecs:', test_vecs.shape)
    print()

    matcher = get_knn_matcher()

    if interactive:

        interactive_demo(mean, eigenvectors, 
                         train_images, train_vecs, train_labels,
                         test_images, test_vecs, test_labels,
                         matcher, knn_k)
    
        return
        
    num_test = len(test_images)

    total_errors = 0

    start = datetime.datetime.now()
    
    print(f'evaluating knn accuracy with k={knn_k}...')

    for start_idx in range(0, num_test, BATCH_SIZE):

        end_idx = min(start_idx + BATCH_SIZE, num_test)
        cur_batch_size = end_idx - start_idx

        idx, labels_pred = match_knn(matcher, knn_k,
                                     test_vecs[start_idx:end_idx],
                                     train_vecs, train_labels)

        labels_true = test_labels[start_idx:end_idx]

        total_errors += (labels_true != labels_pred).sum()
        error_rate = 100.0 * total_errors / end_idx

        print(f'{total_errors:4d} errors after {end_idx:5d} test examples (error rate={error_rate:.2f}%)')

    elapsed = (datetime.datetime.now() - start).total_seconds()
    print(f'total time={elapsed:.2f} seconds ({elapsed/end_idx:.4f}s per image)')

if __name__ == '__main__':
    main()
