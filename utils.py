import numpy as np
import struct

def load_ubyte(images_ubyte, labels_ubyte):
    """
    Load ubyte and convert to numpy array.
    """
    print 'load %s ...'%(images_ubyte)
    with open(images_ubyte, 'rb') as f:
        f.seek(4)
        xnum = f.read(4)
        f.seek(16)
        ubyte = f.read()
    xnum = struct.unpack('>I', xnum)[0]
    assert len(ubyte) / xnum == 784
    data = np.zeros([xnum,784], dtype=np.uint8)
    for i in xrange(xnum):
        for j in xrange(784):
            data[i, j] = struct.unpack('@B', ubyte[j+784*i])[0]
    # data = np.reshape(data, [xnum, 28, 28])
    data = np.reshape(data, [xnum, 28, 28, 1])

    print 'load %s ...'%(labels_ubyte)
    with open(labels_ubyte, 'rb') as f:
        f.seek(4)
        ynum = f.read(4)
        f.seek(8)
        ubyte = f.read()
    ynum = struct.unpack('>I', ynum)[0]
    assert xnum == ynum
    assert len(ubyte) == ynum
    labels = np.zeros([ynum, 1])
    for i in xrange(ynum):
        labels[i] = struct.unpack('@B', ubyte[i])[0]
        # labels.append(struct.unpack('@B', ubyte[i])[0])
    return data, labels
