import numpy as np
import matplotlib.pyplot as plt
from ADP_utils import ordinal, tf_float32, amp_to_db, exp_sigmoid


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, axlen=None, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])


        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        # ax.legend(bars, data.keys(), fancybox=True,  loc='upper center', bbox_to_anchor=(0, 0), ncol=4)
        ax.legend(bars, data.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=4, fontsize=20)
        ax.set_xticks(np.arange(0,axlen,1))



def subplot_input(outputs, duration, channels_pitch, max_freq = 1000):
    a = np.zeros((outputs["conditioning"].shape[1], 127))
    for i in range(a.shape[0]):
        notes = np.unique(outputs["conditioning"][:,i,0]).astype(int)
        notes = notes[notes>0]
        a[i, notes] = 1
    channels = [0,1]
    channels_harm_disp = [0]
    n_overtones = 5
    N = outputs["conditioning"].shape[1]
    factor = 250
    fig = plt.figure(figsize=(14,10))

    ax1 = plt.subplot(211)
    samples = np.linspace(0, len(outputs['magnitudes_2'][0,:,0])/factor, N)
    t = np.linspace(0, duration, N)
    xticks_loc = np.arange(0, N, step=factor)
    xticks_val = np.arange(duration).astype(int)
    yticks_loc = np.arange(24, 127, step=12)
    yticks_val = [f"C{i+1}" for i in range(len(yticks_loc))]
    plt.imshow(a.T, aspect="auto", interpolation="none", cmap="gray_r")
    plt.grid()
    plt.xticks(xticks_loc, [""]*len(xticks_val), fontsize = 18)
    plt.yticks(yticks_loc, yticks_val, fontsize = 18)
    plt.ylim([21, 108])
    #ax.invert_yaxis()
    # ax1.set_title('Input MIDI File')#, rotation='vertical',x=-0.1,y=0.1)
    # ax1.set_xlabel(r"Time in $s$", fontsize = 25)
    plt.xlim([0,N])
    ax1.set_ylabel(r"MIDI Note", fontsize = 25)

    ax2 = plt.subplot(212)
    for i in channels_pitch:
        plt.plot(t, outputs[f'f0_hz_{i}'][0, :, 0], label=fr'$f_0$ channel {i}')
    ax2.set_xlabel(r"Time in $s$", fontsize = 25)
    ax2.set_ylabel(r"Frequency in $Hz$", fontsize = 25)
    plt.xlim(0,duration)
    plt.ylim(0,max_freq)
    plt.legend()
    box = ax2.get_position()
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                fancybox=True, shadow=True, ncol=5,fontsize=20)
    plt.xticks(xticks_val, fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.grid('on')
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.15   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # plt.tight_layout()
    return fig

def subplot_output(outputs, duration, channels_pitch, channels_amp, channel_harm_dist, overtones_harm_dist, max_freq = 1000):

    fig = plt.figure(figsize=(14,10))
    N = outputs["conditioning"].shape[1]
    n_overtones = 5
    factor = 250
    samples = np.linspace(0, len(outputs['magnitudes_2'][0,:,0])/factor, N)
    t = np.linspace(0, duration, N)
    xticks_loc = np.arange(0, N, step=factor)
    xticks_val = np.arange(duration).astype(int)

    ax1 = plt.subplot(311)
    for i in channels_amp:
        plt.plot(t, amp_to_db(exp_sigmoid(outputs[f'amplitudes_{i}'][0, :, 0])), label=f'Amplitude channel {i}')
    plt.xlim(0, duration)
    plt.xticks(xticks_val, [""]*len(xticks_val), fontsize = 18)
    plt.yticks(fontsize=18)

    box = ax1.get_position()
    plt.legend()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title('Channel Amplitudes', fontsize=18)#, rotation='vertical',x=-0.1,y=0.1)
    ax1.set_ylabel(r"Amplitude in $dB$", fontsize=25)
    ax1.grid('on')
    print(len(xticks_loc))
    print(len(xticks_val))

    ax2 = plt.subplot(312)
    for overtone in overtones_harm_dist:
        plt.plot(t, amp_to_db(exp_sigmoid(outputs[f'harmonic_distribution_{channel_harm_dist}'][0, :, overtone])), "-", label=f"Amplitude {ordinal(overtone+1)} harmonic" )
    plt.xlim(0, duration)
    plt.xticks(xticks_val, [""]*len(xticks_val), fontsize = 18)
    plt.yticks(fontsize=18)

    box = ax2.get_position()
    plt.legend()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_title(f'Harmonic distribution of first {n_overtones} Harmonics of Channel {channel_harm_dist}', fontsize=18)#, rotation='vertical',x=-0.1,y=0.1)
    ax2.set_ylabel(r"Amplitude in $dB$", fontsize=25)
    ax2.grid('on')
    print(len(xticks_loc))
    print(len(xticks_val))
 

    noise_mag = [0,1,9,19,29,39]
    n_magnitues =  6
    ax3 = plt.subplot(313)
    for overtone in noise_mag:
        plt.plot(t, amp_to_db(exp_sigmoid(outputs[f'magnitudes_{channel_harm_dist}'][0, :, overtone])), "-", label=f"{ordinal(overtone+1)} Noise Magnitude" )
    #plt.xlabel('time in s')
    plt.xlim(0,duration)
    plt.legend()
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_title(f'Noise distribution of {n_magnitues} Magnitudes of Channel {channel_harm_dist}', fontsize=18)#, rotation='vertical',x=-0.1,y=0.1)
    plt.xticks(xticks_val, fontsize=18)
    plt.yticks(fontsize=18)
    #plt.xticks(xticks_loc, [""]*len(xticks_loc))
    ax3.set_xlabel(r"Time in $s$", fontsize=25)
    ax3.set_ylabel(r"Amplitude in $dB$", fontsize=25)
    plt.grid()
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.85   # the right side of the subplots of the figure
    bottom = 0.15   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    return fig

def plot_context(outputs, duration):
    plt.figure(figsize=(14,10))
    ax = plt.gca()
    N = outputs["conditioning"].shape[1]
    factor = 250
    samples = np.linspace(0, len(outputs['magnitudes_2'][0,:,0])/factor, N)
    t = np.linspace(0, duration, N)
    xticks_loc = np.arange(0, N, step=factor)
    xticks_val = np.arange(duration).astype(int)
    plt.plot(t, outputs['context'][0, :, :])
    plt.xlim(0,duration)
    plt.legend()
    plt.xticks(xticks_val, fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_xlabel(r"Time in $s$", fontsize=25)
    ax.set_ylabel(r"Value", fontsize=25)
    plt.grid()
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.85   # the right side of the subplots of the figure
    bottom = 0.15   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)