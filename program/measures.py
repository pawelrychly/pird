__author__ = 'Pawel Rychly'
import numpy
import numpy.fft
import decimal



def real_fft(q):
    #print q
    f = numpy.fft.rfft(q)
    f = f[1:]
    f =  [numpy.sqrt(value.real**2 + value.imag**2) for value in f]
    return f

def euclidean_distance(c1, c2):
    return numpy.sqrt((c1["x"] - c2["x"])**2 + (c1["y"] - c2["y"])**2 + (c1["z"] - c2["z"])**2)

# Standard
def err_xy(c_vec):
    x_vec = [value["x"] for value in c_vec]
    y_vec = [value["y"] for value in c_vec]
    num_x = len(c_vec) - 2

    if num_x <= 0:
        return None
    x_avg = numpy.mean(x_vec)

    y_avg = numpy.mean(y_vec)

    #s_xy =  numpy.sum([((y_vec[i] - y_avg) * (x_vec[i] - x_avg ))**2 for i in range(len(x_vec))])
    s_xy =  numpy.sum([((y_vec[i] - y_avg) * (x_vec[i] - x_avg )) for i in range(len(x_vec))])

    s_x = numpy.sum([(x_vec[i] - x_avg)**2 for i in range(len(x_vec))])
    s_y = numpy.sum([(y_vec[i] - y_avg)**2 for i in range(len(y_vec))])

    #print s_xy
    #print s_x
    #print s_y
    if s_x != 0:
        return numpy.sqrt((s_y - ((s_xy**2)/s_x))/num_x)
        return numpy.nan #numpy.sqrt((s_y / num_x))

#flattening factor
#evaluates creature movement dynamics in the xy plane
def ff(d_xy):
    d_xy_avg = numpy.mean(d_xy)
    if d_xy_avg != 0:
        return numpy.std(d_xy, ddof=1)/d_xy_avg

#vertical elongation factor
# large value - jumping style of movement
# small value - crawling creatures
#evaluates creature movement dynamics in the xy plane
def vef(d_z):
    d_z_avg = numpy.mean(d_z)
    if d_z_avg != 0:
        return numpy.std(d_z, ddof=1)/d_z_avg

#height to width measure
def hw(d_xy, d_z):
    d_xy_std = numpy.std(d_xy, ddof=1)
    if d_xy_std != 0:
        return numpy.std(d_z, ddof=1)/d_xy_std

#mean speed
def xyz_speed(c_vec):
    length = len(c_vec)
    if length > 0:
        distances = [ euclidean_distance(c_vec[i-1], c_vec[i]) for i in range(1,length)]
        return numpy.sum(distances)/length
    return 0

#Spectral Flatness Measure
# if creature velocity may be presented as a sinusoid then sfm is small
# if it is irregular or uniform then sfm is higher
def sfm(values):
    length = len(values)
    f = real_fft(values)
    f_avg = numpy.mean(f)
    if f_avg != 0:
        ln_f = [numpy.log(value) for value in f]
        ln_f_avg = numpy.mean(ln_f)
        return numpy.exp(ln_f_avg)/f_avg

#The most significant frequency - highest amplitude
def max_f(values):
    length = len(values)
    f = real_fft(values)
    f_max = numpy.argmax(f)

    return f_max


def corrcoef(vec_1, vec_2):
    #print vec_1
    #print vec_2
    covariance = numpy.cov(vec_1, vec_2)

    #print covariance
    correlation_coefficient = [[ covariance[i][j]/numpy.sqrt(covariance[i][i]*covariance[j][j]) for j in range(len(covariance[i]))] for i in range(len(covariance))]
    #print correlation_coefficient
    return correlation_coefficient

#maximal correlation of xyz speed with its offset
def max_f_auto(values):
    #print(c_vec)
    length = len(values)
    auto_corr_size = (length - 1) / 4
    auto_corr = [ 0 for i in range(auto_corr_size)]
    auto_corr[0] = 1

    for i in range(1, auto_corr_size):
        x_v = values[0:len(values)-i]
        y_v = values[i:len(values)]
        if all([ y == y_v[0] for y in y_v]):
            return numpy.nan
        auto_corr[i] = numpy.corrcoef([x_v,y_v])[0][1]
            #auto_corr[i] = corrcoef(speed[0:len(speed)-i], speed[i:len(speed)])[0][1]
    lower_bound = 0
    for i in range(1, auto_corr_size):
        if auto_corr[i] > auto_corr[i-1]:
            lower_bound = i
            break
    max_sig_auto_corr = max(auto_corr[lower_bound: auto_corr_size])
    argmax_sig_auto_corr = numpy.argmax(auto_corr[lower_bound: auto_corr_size])
    return (argmax_sig_auto_corr, max_sig_auto_corr)
