'''
Script generating feature vector from given file in form of a yaml file
It extends some essentia built-in functions to generate some additional informations
'''

import essentia
import essentia.standard
import essentia.streaming
from pylab import *
from essentia.standard import *
import sys
import os
from measures import *


def prepare_directory(save_pic_dir):
    if not os.path.exists(save_pic_dir):
        os.makedirs(save_pic_dir)

out_file = 'y.yaml'
file_dir = '/hdd/gtzan/genres/classical/classical.00080.au'
if len(sys.argv) >= 3:
    file_dir = sys.argv[1]
    out_file = sys.argv[2]
loader = essentia.standard.MonoLoader(filename = file_dir)
audio = loader()
w = Windowing(type = 'hann')
spectrum = Spectrum()  # FFT() would give the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()

mfccs = []
pool = essentia.Pool()

mfccs = []
i = 0
pool = essentia.Pool()
#print "audio length {0}".format(len(audio))
for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    i +=1
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    mfccs.append(mfcc_coeffs)
    pool.add('lowlevel.mfcc', mfcc_coeffs)
    pool.add('lowlevel.mfcc_bands', mfcc_bands)


mfccs = essentia.array(mfccs).T
values_in_time = [mfccs[i,0:] for i in range(0, len(mfccs[:,1]))]
#print len(values_in_time)
results = {"max_f_auto": [], "max_f": [], "sfm":[], "argmaxf_auto": []}

aggrPool = PoolAggregator(defaultStats = [ "mean", "var", "min", "max" ])(pool)
for i in range(len(values_in_time)):
    (argmaxf_auto, maxf_auto) = max_f_auto(values_in_time[i])
    maxf = max_f(values_in_time[i])
    sfm_v = sfm(values_in_time[i])
    if maxf_auto == None:
        maxf_auto = 0
    if maxf == None:
        maxf = 0
    if sfm_v == None:
        sfm_v = 0
    if argmaxf_auto == None:
        argmaxf_auto = 0

    maxf_auto = float(maxf_auto)
    results["max_f_auto"].append(maxf_auto)
    results["argmaxf_auto"].append(argmaxf_auto)
    results["max_f"].append(maxf)
    results["sfm"].append(sfm_v)

aggrPool.add("lowlevel.mfcc.max_f_auto", results["max_f_auto"])
aggrPool.add("lowlevel.mfcc.max_f", results["max_f"])
aggrPool.add("lowlevel.mfcc.argmaxf_auto", results["argmaxf_auto"])
aggrPool.add("lowlevel.mfcc.sfm", results["sfm"])


output = YamlOutput(filename = out_file)
output(aggrPool)


