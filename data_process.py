import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import shelve
import pprint
import pickle


def pload(filename):
    with open(filename, 'rb') as f:
        dat = pickle.load(f, encoding='latin1')
    return dat


def im2vid(path, name='video', frate=24, suffix='png'):
    image_folder = path
    video_name = os.path.join(path, name + '.avi')

    images = [img for img in os.listdir(image_folder) if img.endswith(suffix)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
        'X', 'V', 'I', 'D'), frate, (width, height), True)
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


def savevar(self, path, **vardict):
    data = shelve.open(path)
    data = vardict
    data.close()


def loadvar(self, path):
    data = shelve.open(path)
    return data


def corr(a, b):
    # correlation coefficient between a and b
    # 1d or 2d data are acceptable
    if not a.shape == b.shape:
        print('two array should have the same dimension')

    p = np.mean((a - a.mean()) * (b - b.mean()))
    stds = a.std() * b.std()
    if stds == 0:
        return 0
    else:
        p /= stds
        return p


def acf(x, length=10):
    # 1-dimensional auto-correlation function
    return np.array([1] + [corr(x[:-i], x[i:]) for i in range(1, length)])


def arr_acf(x, length=10):
    # 2d auto-correlation coefficient with time in 0-dimension
    return np.array([1] + [corr(x[:-i, :], x[i:, :]) for i in range(1, length)])


def calc_angle(t1, p1, t2, p2):
    t1 = t1 * np.pi / 180
    t2 = t2 * np.pi / 180
    p1 = p1 * np.pi / 180
    p2 = p2 * np.pi / 180
    v1 = np.array([np.cos(p1) * np.sin(t1), np.cos(p1)
                   * np.cos(t1), np.sin(p1)])
    v2 = np.array([np.cos(p2) * np.sin(t2), np.cos(p2)
                   * np.cos(t2), np.sin(p2)])
    angle = np.arccos(np.dot(v1, v2)) * 180 / np.pi
    return angle


class Batch(object):

    def __init__(self, path, suffix):
        self.filelist = []
        self.folder = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.find(suffix) != -1:
                    self.filelist.append(os.path.join(root, file))
                    self.folder.append(root.replace(path, '..'))
        self.length = len(self.filelist)

    def batch(self, func, *args, **kwargs):
        var = []
        i = 0
        for name in self.filelist:
            dat = pload(name)
            var.append(func(dat, i, *args, **kwargs))
            i += 1
        return var


class folderBatch(object):

    def __init__(self, path, suffix):
        self.list = []
        for root, dirs, files in os.walk(path):
            for subfolder in dirs:
                if subfolder.find(suffix) != -1:
                    self.list.append(os.path.join(root, subfolder))
        self.length = len(self.list)


class ODMR():

    def __init__(self, filename):
        self.name = filename
        self.data = pload(filename)
        self.matrix = self.data['counts_matrix_all']
        self.freq = self.data['frequency']
        self.t = self.data['time']
        self.n, self.w = self.matrix.shape

        self.counts = np.mean(self.matrix, 1)
        if min(self.counts) > 0:
            self.matrix_norm = self.matrix / self.counts.reshape([-1, 1])
        else:
            self.matrix_norm = (self.matrix + 0.001) / \
                (self.counts + 0.001).reshape([-1, 1])

    def pnmat(self, start=0, end=-1, binning=1, saveim=0, **kwargs):

        if end == -1:
            array = self.matrix_norm[start:, :]
            name = self.name.replace('ODMR.pys', 'nmatrix_%d_end.png' % start)
        else:
            array = self.matrix_norm[start:end, :]
            name = self.name.replace(
                'ODMR.pys', 'nmatrix_%d_%d.png' % (start, end))

        if binning != 1:
            l = array.shape[0]
            n = l // binning
            array = np.mean(
                array[0: n * binning, :].reshape([n, binning, -1]), 1)
            name = name.replace('.png', '_bin%d.png' % binning)

        t1 = start * self.data['run_time'] / self.n
        t2 = array.shape[0] * binning * self.data['run_time'] / self.n + t1
        asp = array.shape[1] / array.shape[0]

        if saveim == 2:
            plt.imsave(fname=name, arr=np.transpose(
                array), **kwargs)
        else:
            plt.imshow(np.transpose(array), extent=(
                t1, t2, self.freq[0] / 1e6, self.freq[-1] / 1e6), aspect=asp, ** kwargs)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (MHz)')
            if saveim == 1:
                plt.savefig(name, dpi=200)
            elif saveim == 0:
                plt.show()

    def pline(self, start=0, end=-1, norm=True, saveim=True):
        if end == -1:
            array = self.matrix_norm[start:, :]
            name = self.name.replace('ODMR.pys', 'ODMR_%d_end.png' % start)
        else:
            array = self.matrix_norm[start:end, :]
            name = self.name.replace(
                'ODMR.pys', 'ODMR_%d_%d.png' % (start, end))

        spec = np.sum(array, 0)
        a = np.sum(array, 0)
        if norm:
            spec = spec / np.max(spec)

        plt.plot(self.freq / 1e6, spec)
        plt.xlabel('Frequency (MHz)')
        if norm:
            plt.ylabel('Normalized intemsity', )
        else:
            plt.ylabel('PL counts (a.u.)')

        if saveim:
            plt.savefig(name, dpi=200)
            plt.close()
        else:
            plt.show()

    def pacf(self, length=10):
        # plot autocorrelation function of counts and normalized ODMR matrix
        name = self.name.replace('ODMR.pys', 'acf%d.png' % length)
        ac1 = acf(self.counts, length)
        ac2 = np.zeros([length, ])
        for i in range(self.w):
            ac2 = np.sum([ac2, acf(self.matrix_norm[:, i], length)], 0)
        ac2 /= self.w

        plt.subplot(2, 1, 1)
        plt.plot(ac1, 'o-')
        plt.title('autocorrelation of counts')
        plt.ylabel('acf of counts')

        plt.subplot(2, 1, 2)
        plt.plot(ac2, '.-')
        # plt.title('autocorrelation of matrix')
        plt.xlabel('time lag')
        plt.ylabel('acf of matrix')

        plt.savefig(name, dpi=200)
        plt.close()

    def parr_acf(self, length=20):
        # plot autocorrelation function of counts and normalized ODMR matrix
        name = self.name.replace('ODMR.pys', 'arr_acf%d.png' % length)
        ac1 = acf(self.counts, length)
        ac2 = np.zeros([length, ])
        ac2 = arr_acf(self.matrix_norm, length)

        plt.subplot(2, 1, 1)
        plt.plot(ac1, 'o-')
        plt.title('autocorrelation of counts')
        plt.ylabel('acf of counts')

        plt.subplot(2, 1, 2)
        plt.plot(ac2, '.-')
        # plt.title('autocorrelation of matrix')
        plt.xlabel('time lag')
        plt.ylabel('acf of matrix')

        plt.savefig(name, dpi=200)
        plt.close()

    def pcf(self, saveim=False):
        # plot cross correlation function of counts and normalized ODMR matrix
        name = self.name.replace('ODMR.pys', 'cf.png')
        cf = np.zeros([self.w, ])
        for i in range(self.w):
            cf[i] = np.corrcoef(self.counts, self.matrix_norm[:, i])[0, 1]

        plt.plot(self.freq, cf)
        if saveim:
            plt.savefig(name, dpi=200)
            plt.close()
        else:
            plt.show()

    def ptcf(self, length=10, saveim=False):
        # plot time lapse correlation function of counts and normalized ODMR matrix
        name = self.name.replace('ODMR.pys', 'tcf%d.png' % length)
        le = self.n - length + 1
        tcf = np.zeros(le)
        for i in range(le):
            tcf[i] = np.sum([np.corrcoef(self.counts[i:i + length],
                                         self.matrix_norm[i:i + length, j])[0, 1] for j in range(self.w)]) / self.w
        plt.plot(tcf)

        if saveim:
            plt.savefig(name, dpi=200)
            plt.close()
        else:
            plt.show()

    def exp_mat(self, path='none'):
        if path == 'none':
            name = self.name.replace('ODMR.pys', 'mat.txt')
        else:
            name = path
        dat = np.vstack(np.array([self.freq, self.matrix]))
        np.savetxt(name, dat, delimiter='\t', fmt='%.1f',
                   header='frequency matrix')

    def exp_line(self, norm=False, path='none'):
        if path == 'none':
            name = self.name.replace('ODMR.pys', 'line.txt')
        else:
            name = path
        if norm == True:
            temp = np.sum(self.matrix, 0)
            dat = np.vstack(np.array([self.freq / 1e6, temp / np.max(temp)])).T
        else:
            dat = np.vstack(
                np.array([self.freq / 1e6, np.sum(self.matrix, 0)])).T
        np.savetxt(name, dat, delimiter='\t', fmt='%g',
                   header='frequency (MHz), counts')


class Trace():
    def __init__(self, filename):
        self.name = filename
        self.data = pload(filename)
        self.t = self.data['CT']
        self.x = self.data['CposX']
        self.y = self.data['CposY']
        self.z = self.data['CposZ']

    def export(self, path='none'):
        if path == 'none':
            name = self.name.replace('pys', 'txt')
        else:
            name = path
        dat = np.transpose(np.array([self.t, self.x, self.y, self.z]))
        np.savetxt(name, dat, delimiter='\t',
                   fmt='%d', header='T\tX\tY\tZ')


# class Image():
#     def __init__(self, filename):
#         self.name = filename
#         self.data = pload(filename)
#         self.time = self.data['finish_time']


if __name__ == '__main__':
    # b1=Batch(r'K:\ODMR\2018-02-08 ND counts\nv1600 4.6uw', '_trace.pys')
    # b2=Batch(r'K:\ODMR\Tracking\2018-02-06 cell - Copy\cell1\2 ND1', 'ODMR.pys')
    # m=b1.batch(lambda x, i: np.mean(x['Ccounts'][-51:-1]))
    # a = ODMR(r'K:\ODMR\Tracking\2018-03-09\1 cell1\6 ND1_ODMR.pys')
    # a.pline(saveim=True, start=3250, end=3260)
    # a.pnmat(start=0, binning=10, saveim=0)

    path = r'K:\ODMR\Tracking\2018-06-12 Hela\1 cell1'

    b = Batch(path, 'ODMR.pys')
    for name in b.filelist:
        pprint.pprint(name)
        a = ODMR(name)
        a.pnmat(saveim=2)
        # a.exp_line(norm=False)
        # a.pline(saveim=True, start=3250, end=3260)
        # a.exp_mat()

    f = folderBatch(path, 'timelapse')
    for folder in f.list:
        pprint.pprint(folder)
        im2vid(folder, frate=10)
