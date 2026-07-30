'''
This script contain the basic plotting functions and plotting parameters (e.g., color, font, etc.) for the figures.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpmath import *
import math
import matplotlib.transforms
import matplotlib.path
from matplotlib.collections import LineCollection
import matplotlib.path
import matplotlib.patches as patches
import colorsys
import random

#------------------- fonts
plt.rcParams["axes.edgecolor"] = "k"
plt.rcParams["axes.facecolor"] = "w"
plt.rcParams["axes.linewidth"] = "1"
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
#plt.rc('text', usetex=True )
#plt.rc('font', family='times new roman', weight='normal', size=14)
#plt.rcParams["mathtext.fontset"] = "custom"
#plt.rcParams["mathtext.rm"] = "times new roman"
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({"font.family": "Helvetica"})
#plt.rcParams["mathtext.fontset"] = "stix"
'''
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({"font.family": "times new roman"})
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "times new roman"
plt.rcParams["mathtext.it"] = "times new roman italic"
#plt.rcParams["font.sans-serif"] = "Helvetica"
#plt.rcParams["mathtext.fontset"] = "Helvetica"
'''
plt.rcParams['pdf.fonttype'] = 42 # prepare as vector graphic
plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['text.usetex'] = True
#https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
alphFont = 12 # font size for figure alphabets
titleFont = 10 # font size of figure titles

#------------------- colors

#cK[k][L] gives the color that corresponds to k, L case
cK = {1: {8: '#806934', 32: '#BF9D4E', 128: '#FFD269'},
      2: {8: '#462352', 32: '#A251BD', 128: '#D86DFC'},
      3: {8: '#265C36', 32: '#409C5B', 128: '#5ADB81'}}

cBP = 'gray' # mean-field BP color
cExponent = 'gray' # color of exponent line for avalanches

#cR = {8: '#344B80', 32: '#4E70BF', 128: '#6997FF'}
#cR = {8: '#313F66', 32: '#566EB3', 128: '#7A9EFF'}

  
cR = cK[1]
# colors for rewiring
# cRWL1 = '#344B80'
# cRWL2 = '#4E70BF'
# cRWL3 = '#6997FF'

#------------------- legend properties
hL = 1.7 # legend handle length
hPad = 0.5 # legend handle textpad
ls = '-'

#------------------- axis, figure size properties (in cm)
fig_width_2col = 2*8.6
fig_width_1col = 8.6

avsize_lable = r'$S$'
prob_lable = r'$P(S)$'


#------------------ others
lw = 2 # line width

#------------------ utils functions
# To create custom color map
#https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
def rgb_to_hex(r,g,b, alpha):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_inverse_dict(d): # https://stackoverflow.com/questions/35491223/inverting-a-dictionary-with-list-values
    inverse = {}
    for k,v in d.items():
        if isinstance(v, list):
            for x in v:
                inverse.setdefault(x, []).append(k)
        else:
            inverse.setdefault(v, []).append(k)
        
            
    return inverse

def get_qualitatively_different_colors(n_colors, l_default = 0.5, s_default = 0.5, shuffle_state=None, init_h=0):
    '''
    Return a list of size n_colors with the most possible different rgb colors. The lightness and saturation are mantained constant.
    '''
    cmap = [colorsys.hls_to_rgb(h/360, l_default, s_default) for h in np.linspace(init_h, init_h+360, num=n_colors, endpoint=False)]
    if shuffle_state is not None:
        random.Random(shuffle_state).shuffle(cmap)
    return cmap

def get_unicolormap(rgb_color_tuple, n_colors, min_l=0.2, max_l=0.9):
    '''
    Gets a colormap of size n_colors by changing the lightness of a specific color (so monochromatic colormap). The min_l and max_l parameters refer to
    minimum and maximum lightness, because if we want to use this unicolormap in same figure with other unicolormaps we don't want it to get black or white
    because then we won't be able to distinguish.
    '''
    if min_l > 1 or min_l < 0 or max_l > 1 or min_l < 0:
        print(f"Minimum and maximum luminance values must lie between [0,1] but min_l: {min_l} and max_l:{max_l} were given.")
        # This is needed because apparently colorsys only work with values between [0, 1] BUT allow for higher values and gives weird behaviour
        raise ValueError
    base_h, base_l, base_s = colorsys.rgb_to_hls(*rgb_color_tuple)
    if n_colors == 1:
        cmap = [colorsys.hls_to_rgb(base_h, base_l, base_s)]
    elif n_colors == 2:
        cmap = [colorsys.hls_to_rgb(base_h, base_l, base_s), colorsys.hls_to_rgb(base_h, max_l, base_s)]
    else:
        cmap = [colorsys.hls_to_rgb(base_h, l, base_s) for l in np.linspace(min_l, max_l, num=n_colors, endpoint=False)]
    return cmap

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def coloredArrow(ax, start, end, cmap, n=50, lw=3, axiscoord=True):
    '''https://stackoverflow.com/questions/47163796/using-colormap-with-annotate-arrow-in-matplotlib'''
    
    cmap = plt.get_cmap(cmap,n)
    # Arrow shaft: LineCollection
    x = np.linspace(start[0],end[0],n)
    y = np.linspace(start[1],end[1],n)
    x_back = np.linspace(start[0]-0.05,end[0]+0.05,n)
    y_back = np.linspace(start[1],end[1],n)
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1],points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0,1,n))
    if not axiscoord:
        lc.set_clip_on(False)
    ax.add_collection(lc)

    # Arrow head: Triangle
    tricoords = [(0,-0.4),(0.5,0),(0,0.4),(0,-0.4)]
    angle = np.arctan2(end[1]-start[1],end[0]-start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(end[0],end[1], c=1, s=(2*lw)**2, marker=tri, cmap=cmap,vmin=0, clip_on=False)
    #ax.autoscale_view()

def split_arrow(arrow, color_tail="C0",color_head="C0", ls_tail="-", ls_head="-",lw_tail=1.5, lw_head=1.5):    
        v1 = arrow.get_path().vertices[0:3,:]
        c1 = arrow.get_path().codes[0:3]
        p1 = matplotlib.path.Path(v1,c1)
        pp1 = patches.PathPatch(p1, color=color_tail, linestyle=ls_tail, 
                                fill=False, lw=lw_tail)
        pp1.set_clip_on(False)
        arrow.axes.add_patch(pp1)

        v2 = arrow.get_path().vertices[3:8,:]
        c2 = arrow.get_path().codes[3:8]
        c2[0] = 1
        p2 = matplotlib.path.Path(v2,c2)
        pp2 = patches.PathPatch(p2, color=color_head, lw=lw_head, linestyle=ls_head)
        pp2.set_clip_on(False)
        arrow.axes.add_patch(pp2)
        arrow.remove()

def get_color_shades(color, ncolors=3, sat_inc=0.1, lum_inc=0.0):
    """
    Creates a set of color shades based on a given color
    color: the color, in any format that matplotlib can recognize
    ncolors: number of colors to generate
    sat_inc: change of saturation in each color. Can be negative
    lum_inc: change of luminance in each color. Can be negative.
    """
    rgb = mcolors.to_rgb(color)
    hsv = mcolors.rgb_to_hsv(rgb)
    
    new_colors = np.empty((ncolors,3))
    
    for j in range(ncolors):
        saturation = np.clip(hsv[1] + sat_inc*j, 0.0, 1.0)
        luminance =  np.clip(hsv[2] + sat_inc*j, 0.0, 1.0)
        new_colors[j] = mcolors.hsv_to_rgb([hsv[0], saturation, luminance])
        
    
    return new_colors

def normal_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
def to_inches(cm):
    """
    Convert cm to inches
    """
    return cm/2.54

def one_col_fig(height=fig_width_1col/1.618):
    """
    Returns a tuple (w,h), where w is the width if a single-column graph.
    Height by default is the golden ratio of the width, but can be chosen (in cm)
    """
    width = to_inches(fig_width_1col)
    height = to_inches(height)
    return (width, height)

def two_col_fig(height=fig_width_2col/1.618):
    """
    Returns a tuple (w,h), where w is the width if a double-column graph.
    Height by default is the golden ratio of the width, but can be chosen (in cm)
    """
    width = to_inches(fig_width_2col)
    height = to_inches(height)
    return (width, height)

def make_unvisible(ax, transparent=False):
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    #ax.axis('off')
    if transparent:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
def arrowed_spines(ax, x_width_fraction=0.05, x_height_fraction=0.05, lw=None, ohg=0.3, locations=('bottom right', 'left up'), **arrow_kwargs):
# Found here https://stackoverflow.com/questions/33737736/matplotlib-axis-arrow-tip
    """
    Add arrows to the requested spines
    Code originally sourced here: https://3diagramsperpage.wordpress.com/2014/05/25/arrowheads-for-axis-in-matplotlib/
    And interpreted here by @Julien Spronck: https://stackoverflow.com/a/33738359/1474448
    Then corrected and adapted by me for more general applications.
    :param ax: The axis being modified
    :param x_{height,width}_fraction: The fraction of the **x** axis range used for the arrow height and width
    :param lw: Linewidth. If not supplied, default behaviour is to use the value on the current left spine.
    :param ohg: Overhang fraction for the arrow.
    :param locations: Iterable of strings, each of which has the format "<spine> <direction>". These must be orthogonal
    (e.g. "left left" will result in an error). Can specify as many valid strings as required.
    :param arrow_kwargs: Passed to ax.arrow()
    :return: Dictionary of FancyArrow objects, keyed by the location strings.
    """
    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault('overhang', ohg)
    arrow_kwargs.setdefault('clip_on', False)
    arrow_kwargs.update({'length_includes_head': True})

    # axis line width
    if lw is None:
        # FIXME: does this still work if the left spine has been deleted?
        lw = ax.spines['left'].get_linewidth()

    annots = {}

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = x_width_fraction * (ymax-ymin)
    hl = x_height_fraction * (xmax-xmin)

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(' ')
        assert side in {'top', 'bottom', 'left', 'right'}, "Unsupported side"
        assert direction in {'up', 'down', 'left', 'right'}, "Unsupported direction"

        if side in {'bottom', 'top'}:
            if direction in {'up', 'down'}:
                raise ValueError("Only left/right arrows supported on the bottom and top")

            dy = 0
            head_width = hw
            head_length = hl

            y = ymin if side == 'bottom' else ymax

            if direction == 'right':
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax

        else:
            if direction in {'left', 'right'}:
                raise ValueError("Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl

            x = xmin if side == 'left' else xmax

            if direction == 'up':
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax


        annots[loc_str] = ax.arrow(x, y, dx, dy, fc='k', ec='k', lw = lw,
                 head_width=head_width, head_length=head_length, transform=ax.transAxes, **arrow_kwargs)

    return annots
def arrowed_spines2(fig, ax):
    # Found here https://stackoverflow.com/questions/33737736/matplotlib-axis-arrow-tip
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()
    print((xmin, xmax, ymin, ymax))
    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([]) # labels 
    plt.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False, transform=ax.transAxes) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False, transform=ax.transAxes)
def despine(axs):
    """
    Despine both a single axe or an array of axes
    """
    
    if type(axs) is np.ndarray:
        for ax in np.ravel(axs):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    else:
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
    

#------------------ plotting functions
def MLE_powerlaw_1Dgridsearch(data, alpha_range = np.arange(0.01,2.2,0.01), xmin = 10, xmax = 10**2,\
                              plot_it = 0, cFit= None, label_fit= None):
    """Estimate power-law exponent with MLE.

    Parameters
    -----------
    data : 1d array
        avalanche size or time distribution.
    alpha_range : 1d array
        range for the grid search of exponent.
    xmin: int
        minimum avalanche size for fitting the power law.
    xmax : int
        maximum avalanche size for fitting the power law.
    plot_it, cFit, label_fit : 
        parameters for plotting the log-likelihood versis alpha
       
    Returns
    -------
    best_alpha : float
        Estimated power-law exponent.  
    MLLE: float
        Maximum log-likelihood
    """
    
    
    def L(alpha,xmin,xmax,data):
        n = len(data)
        return -n * ln(zeta(alpha,xmin,0,method = 'borwein')- zeta(alpha,xmax,0,method = 'borwein'))- alpha*np.sum(np.log(data))

    data = data[data <= xmax]
    data = data[(data>=xmin)]
    log_like = np.array([L(k,xmin,xmax,data) for k in alpha_range])
    best_alpha = alpha_range[np.where(log_like == max(log_like))[0]]
    
    if plot_it:
        plt.plot(alpha_range,log_like,linewidth =2, color = cFit, label = label_fit)
        plt.xlabel(r'$\alpha$')
        plt.ylabel('log-likelihood')
    return best_alpha, max(log_like)

    data = data[data <= xmax]
    data = data[(data>=xmin)]
    log_like = np.array([L(k,xmin,xmax,data) for k in alpha_range])
    best_alpha = alpha_range[np.where(log_like == max(log_like))[0]]
    
    if plot_it:
        plt.plot(alpha_range,log_like,linewidth =2, color = cFit, label = label_fit)
        plt.xlabel(r'$\alpha$')
        plt.ylabel('log-likelihood')
    return best_alpha, max(log_like)


def xmaxRatio(av, ratio):
    """ 
    compute maximum avalanche size based on the given ratio of distribution.
    """
    data_list = np.ndarray.tolist(av)
    uniq_list = np.sort(data_list)

    N = int(len(uniq_list)*(ratio))-1
    xmax =int(uniq_list[N])
    return xmax



def plot_av(ax, av, minlogbin = 9, maxlogbin = 10**8, nlogbings = 70, col = 'k', lw = 2, ls = '-', label ='av',\
            plot_fit = 0, fit_zeroValue = 1, ls_powerlaw = '-', \
            alpha_range = np.arange(0.01,2.2,0.01), xmin = 10, xmax = 10**2):
    """Plot avalanche size/time duration.

    Parameters
    -----------
    ax : object
        plotting axis.
    av : 1d array
        avalanche size or time distribution.
    minlogbin : int
        minimum avalanche size to start log binning.
    maxlogbin : int
        maximum avalanche size for log binning.
    nlogbings : int
        number of log bins for histogram. 
    col, lw, ls, label : 
        plotting parameters
    plot_fit: boealian
        If plot the power-law fit line.
    fit_zeroValue: float
        The value of fitted power-law line at avalanche-szie = 1.
    ls_powerlaw: string
        Line style for power-law line
    alpha_range, xmin, xmax:
        Parameters for fitting avalanches with power law. You can set the xmax using the percentiles of distribution.
       
    Returns
    -------
    alpha : float
        Estimated power-law exponent.     
    """
    # log binning for large avalanches
    logmin = np.log10(minlogbin)
    logmax = np.log10(maxlogbin)
    bins_log = np.logspace(logmin, logmax, nlogbings)
    bins_log = bins_log[bins_log>minlogbin]

    # linear binning for small avalanches
    minx, maxx, nx = 1,minlogbin,1
    bins_lin = np.arange(minx, maxx, nx)

    bins = np.concatenate((bins_lin,bins_log))
    dist = np.histogram(av, bins=bins, density=True)[0]
    ax.loglog(bins[:-1][dist>0], dist[dist>0], color=col, lw=lw, ls = ls, label=label)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    if plot_fit:
        alpha, MLLE = MLE_powerlaw_1Dgridsearch(av, alpha_range, xmin, xmax)
        ax.plot(bins[:-1], fit_zeroValue*bins[:-1]**(-alpha), color=cExponent, lw=1, ls = ls_powerlaw)
        return alpha[0]
    else: return None


