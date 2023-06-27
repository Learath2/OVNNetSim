import itertools
from enum import IntEnum, auto
from pathlib import Path
import random
from typing import Callable
from core.network import Network, Connection, calculate_bit_rate, Transceiver, TRANSCEIVER_COLORS
from core.parameters import INITIAL_SIGNAL_POWER

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation

from core.utility import db2lin

g_network: Network | None = None

class Views(IntEnum):
    NETWORK = 0
    CONN_STATS = auto()
    MISC = auto()
    NUM_VIEWS = auto()

VIEW_NAMES = {Views.NETWORK: "Network Topology", Views.CONN_STATS: "Connection Statistics", Views.MISC: "Misc"}

g_current_view: Views = Views.NETWORK
g_current_view_bindhandler: Callable[[KeyEvent], None] | None = None
g_current_opt_target = 'latency'
g_current_network_statefulness = False
g_conns = None
g_current_network_transceivers = Transceiver.FIXED_RATE
g_hide_failed = False

def on_key_press(event: KeyEvent):
    global g_current_view
    old_view = g_current_view

    if event.key == "right":
        g_current_view += 1
    elif event.key == "left":
        g_current_view -= 1

    if g_current_view != old_view:
        g_current_view %= Views.NUM_VIEWS
        render_current_view()
        return

    if g_current_view_bindhandler is not None:
        g_current_view_bindhandler(event)


def generate_random_conns(network: Network, n: int) -> list[Connection]:
    return [network.create_connection(*random.sample(list(network.nodes()), k = 2)) for _ in range(100)]


def render_network_view(fig: Figure):
    global g_network, g_current_view_bindhandler
    g_current_view_bindhandler = None

    if g_network is None:
        return

    ax = fig.add_subplot()
    g_network.draw(ax)


def conn_stats_bindhandler(event: KeyEvent):
    global g_conns, g_network, g_current_opt_target, g_current_network_statefulness, g_current_network_transceivers
    if g_network is None:
        return

    state_changed = False
    if event.key == "ctrl+r":
        g_conns = generate_random_conns(g_network, 100)
        state_changed = True
    elif event.key == "ctrl+t":
        if g_current_opt_target == 'latency':
            g_current_opt_target = 'snr'
        elif g_current_opt_target == 'snr':
            g_current_opt_target = 'latency'
        state_changed = True
    elif event.key == "ctrl+k":
        g_current_network_statefulness = not g_current_network_statefulness
        state_changed = True
    elif event.key == "ctrl+1":
        g_current_network_transceivers = Transceiver.FIXED_RATE
        state_changed = True
    elif event.key == "ctrl+2":
        g_current_network_transceivers = Transceiver.FLEX_RATE
        state_changed = True
    elif event.key == "ctrl+3":
        g_current_network_transceivers = Transceiver.SHANNON
        state_changed = True
    elif event.key == "ctrl+y":
        g_hide_failed = not g_hide_failed
        state_changed = True

    if state_changed:
        g_network.set_transceivers(g_current_network_transceivers)
        render_current_view()


def render_conn_stats_view(fig: Figure):
    global g_current_view_bindhandler, g_hide_failed
    if g_network is None:
        return

    g_current_view_bindhandler = conn_stats_bindhandler

    conns_streamed = g_network.stream(g_conns, g_current_opt_target, g_current_network_statefulness)
    ax = fig.subplot_mosaic("ABC")

    if g_hide_failed:
        conns_streamed = [c for c in conns_streamed if not c.failed()]

    snrs = [c.snr() for c in conns_streamed]
    ax['A'].hist(snrs, edgecolor='white')
    ax['A'].set_xlabel("SNR (dB)")

    HIST_LAT_MAX=10
    bins = np.arange(0, HIST_LAT_MAX, 0.5)
    lats = np.clip(np.array([c.latency() for c in conns_streamed]) * 1e3, bins[0], bins[-1])
    ax['B'].hist(lats, bins=bins, edgecolor='white')
    ax['B'].set_xlabel("Latency (ms)")

    bitrates = [c.bitrate() for c in conns_streamed]
    ax['C'].hist(bitrates, edgecolor='white')
    ax['C'].set_xlabel("Bitrate Gbps")
    ax['C'].axhline(np.mean(bitrates), linestyle="--")


def render_misc_view(fig: Figure):
    global g_current_view_bindhandler
    g_current_view_bindhandler = None

    ax = fig.add_subplot()
    snrs = np.linspace(0, 50, 1000)

    for strategy in Transceiver:
        mf = lambda x: calculate_bit_rate(x, strategy)
        vf = np.vectorize(mf)
        brs = vf(db2lin(snrs))

        ax.plot(snrs, brs, linewidth=2, label=str(strategy), color=TRANSCEIVER_COLORS[strategy])
        ax.set_xlabel('GSNR (dB)')
        ax.set_ylabel('Bit Rate (Gbps)')
        ax.set_yscale('log')
        ax.legend()


def get_current_view_state():
    global g_current_view, g_current_opt_target, g_current_network_statefulness, g_current_network_transceivers, g_hide_failed
    match g_current_view:
        case Views.CONN_STATS:
            return f"        Optimization target: {g_current_opt_target}    Stateful?: {g_current_network_statefulness}    Transceivers: {g_current_network_transceivers}    Hide Failed: {g_hide_failed}"
        case _:
            return ""


def render_current_view():
    fig = plt.figure(0)
    fig.clear()
    match g_current_view:
        case Views.NETWORK:
            render_network_view(fig)
        case Views.CONN_STATS:
            render_conn_stats_view(fig)
        case Views.MISC:
            render_misc_view(fig)
        case _:
            raise Exception("Unknown View")

    offset = ScaledTranslation(20/72., -20/72., fig.dpi_scale_trans)
    t = fig.transFigure + offset
    fig.text(0, 1, VIEW_NAMES[g_current_view] + get_current_view_state(), va='top', ha='left', transform=t)

    fig.canvas.draw()


def main():
    file = Path(input("Path to network spec: "))

    global g_network
    g_network = network = Network(file)

    df = network.weighted_paths(INITIAL_SIGNAL_POWER)
    print(df)

    global g_conns
    g_conns = generate_random_conns(network, 100)

    fig = plt.figure(0)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    render_current_view()
    plt.show()


if __name__ == "__main__":
    main()
