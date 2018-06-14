import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import shelve
import pprint
import pickle
from chaco.api import ArrayPlotData, Plot, Spectral, PlotLabel
from traits.api import HasTraits, Trait, Instance, Property, String, Range, Float, Int, Bool, Array, Enum, Button, on_trait_change, cached_property, List
from traitsui.api import View, Item, HGroup, VGroup, VSplit, Tabbed, EnumEditor, TextEditor, Group
from enable.api import Component, ComponentEditor


def pload(filename):
    with open(filename, 'rb') as f:
        dat = pickle.load(f, encoding='latin1')
    return dat


def savevar(self, path, **vardict):
    data = shelve.open(path)
    data = vardict
    data.close()


def loadvar(self, path):
    data = shelve.open(path)
    return data


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


class ODMR(HasTraits):

    line_label = Instance(PlotLabel)
    line_data = Instance(ArrayPlotData)
    matrix_data = Instance(ArrayPlotData)
    line_plot = Instance(Plot, editor=ComponentEditor())
    matrix_plot = Instance(Plot, editor=ComponentEditor())

    line_start = Int(value=0, label='start', auto_set=False, enter_set=True)
    line_end = Int(value=1, label='end', auto_set=False, enter_set=True)
    matrix_start = Int(value=0, label='start', auto_set=False, enter_set=True)
    matrix_end = Int(value=1, label='end', auto_set=False, enter_set=True)
    binning = Int(value=1, label='binning', auto_set=False, enter_set=True)

    def __init__(self, filename):
        self.name = filename
        self.data = pload(filename)
        self.freq = self.data['frequency']
        self.t = self.data['run_time']
        self.matrix = self.data['counts_matrix_all']
        self.counts = np.mean(self.matrix, 1)
        self.nmat = self.normalize(self.matrix, 1)
        self.binning=1
        self._creat_matrix_plot()
        self._creat_line_plot()
        self.on_trait_change(self._update_matrix_data_value, [
                             'matrix_start', 'matrix_end', 'binning'], dispatch='ui')
        self.on_trait_change(self._update_line_data_value, [
                             'line_start', 'line_end'], dispatch='ui')
        print(self.binning)
        
    def normalize(self, array, bi=1):
    	l = array.shape[0]
        n=l//bi
        array = np.sum(array[0: n * bi, :].reshape([n, bi, -1]), 1)
    	counts=np.mean(array,1)

        if min(counts) > 0:
            nmat = array / counts.reshape([-1, 1])
        else:
            nmat = (array + 0.001) / \
                (counts + 0.001).reshape([-1, 1])
        return nmat

    def _creat_matrix_plot(self):
        matrix_data = ArrayPlotData(image=np.transpose(self.nmat))
        matrix_plot = Plot(matrix_data, padding=8,
                           padding_left=64, padding_bottom=32)
        matrix_plot.index_axis.title = 'Time'
        matrix_plot.value_axis.title = 'Frequence'
        matrix_plot.img_plot('image')
                             # xbounds=(0, self.t),
                             # ybounds=(self.freq[0], self.freq[-1]),
                             # colormap=Spectral)
        self.matrix_data = matrix_data
        self.matrix_plot = matrix_plot

    def _creat_line_plot(self):
        line_data = ArrayPlotData(
            frequency=self.freq, counts=np.mean(self.matrix, 0))
        line_plot = Plot(line_data, padding=8,
                         padding_left=64, padding_bottom=32)
        line_plot.plot(('frequency', 'counts'), style='line', color='blue')
        line_plot.index_axis.title = 'Frequency [MHz]'
        line_plot.value_axis.title = 'Fluorescence counts'
        line_label = PlotLabel(text='', hjustify='left',
                               vjustify='bottom', position=[64, 128])
        line_plot.overlays.append(line_label)
        self.line_label = line_label
        self.line_data = line_data
        self.line_plot = line_plot

    def _update_matrix_data_value(self):
        if self.matrix_end == -1:
            array = self.matrix[self.matrix_start:, :]
        else:
            array = self.matrix[self.matrix_start:self.matrix_end, :]

        array=self.normalize(array, self.binning)
        self.matrix_data.set_data('image', np.transpose(array))

    def _update_line_data_value(self):
        if self.line_end == -1:
            array = self.matrix[self.line_start:, :]
        else:
            array = self.matrix[self.line_start:self.line_end, :]

        self.line_data.set_data('counts', np.mean(array, 0))

    def demo(self):
        pass

    def pnmat(self, start=0, end=-1, binning=1, saveim=False, **kwargs):
        name = self.name.replace('ODMR.pys', 'nmatrix.png')
        if end == -1:
            array = self.nmat[start:, :]
        else:
            array = self.nmat[start:end, :]

        if binning != 1:
            l = array.shape[0]
            n = l // binning
            array = np.mean(
                array[0: n * binning, :].reshape([n, binning, -1]), 1)
            name = name.replace('nmatrix.png', 'nmatrix_bin%d.png' % binning)

        if saveim:
            plt.imsave(fname=name, arr=np.transpose(
                array), **kwargs)
        else:
            plt.imshow(np.transpose(array), **kwargs)
            plt.show()

    def plotline(self, start=0, end=-1, binning=1, saveim=False, **kwargs):
        name = self.name.replace('ODMR.pys', 'line.png')

    traits_view = View(VGroup(HGroup(Item('matrix_start'),
                                     Item('matrix_end'),
                                     Item('binning'),
                                     ),
                              Item('matrix_plot', resizable=True, show_label=False),
                              HGroup(Item('line_start'),
                                     Item('line_end'),
                                     ),
                              Item('line_plot', resizable=True, show_label=False),
                              ),
                       width=500, height=500, resizable=True
                       )


# class trace():
#     def __init__():


if __name__ == '__main__':
    # b1 = Batch(r'K:\ODMR\2018-02-08 ND counts\nv1600 4.6uw', '_trace.pys')
    # b2 = Batch(r'K:\ODMR\Tracking\2018-02-06 cell - Copy\cell1\2 ND1', 'ODMR.pys')
    # m = b1.batch(lambda x, i: np.mean(x['Ccounts'][-51:-1]))
    d1 = ODMR(r'D:\Data\2018\2018-03-09\2 cell2\3 ND1_ODMR.pys')
    d1.configure_traits()
