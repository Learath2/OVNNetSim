from copy import deepcopy
from enum import StrEnum
import itertools
import json
import math
from pathlib import Path
from collections import deque
from collections.abc import Iterable
from typing import Literal, NotRequired, Self, Tuple, TypedDict
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.markers as mplm
import matplotlib.collections as mcoll
import matplotlib.transforms as mtrans
import matplotlib.patches as mpatches
import matplotlib.text as mtext

from scipy.special import erfcinv
from scipy.constants import pi, h

from core.utility import Point2D, db2lin, euc_dist, lin2db
from core.parameters import AMPLIFIER_GAIN, C_IN_FIBER, BER_THRESHOLD, FIBER_LOSS_COEFF, RS, BN, INITIAL_SIGNAL_POWER, CBAND_CENTER, AMPLIFIER_NOISE

class Transceiver(StrEnum):
    FIXED_RATE = 'fixed-rate'
    FLEX_RATE = 'flex-rate'
    SHANNON = 'shannon'

TRANSCEIVER_COLORS = { Transceiver.FIXED_RATE: 'orangered', Transceiver.FLEX_RATE: 'deepskyblue', Transceiver.SHANNON: 'greenyellow'}

def calculate_bit_rate(gsnr: float, strategy: Transceiver):
    match strategy:
        case Transceiver.FIXED_RATE:
            if gsnr >= 2 * (erfcinv(2 * BER_THRESHOLD) ** 2) * (RS / BN):
                return 100 #PM-QPSK
            else:
                return 0
        case Transceiver.FLEX_RATE:
            if gsnr < 2 * (erfcinv(2 * BER_THRESHOLD) ** 2) * (RS / BN):
                return 0
            elif gsnr < (14/3) * (erfcinv((3/2) * BER_THRESHOLD) ** 2) * (RS / BN):
                return 100 #PM-QPSK
            elif gsnr < 10 * (erfcinv((8/3) * BER_THRESHOLD) ** 2) * (RS / BN):
                return 200 #PM-8QAM
            else:
                return 400 #PM-16QAM
        case Transceiver.SHANNON:
            return 2 * RS * np.log2(1 + gsnr * (RS / BN)) * 1e-9
        case _:
            raise Exception("unreachable")

class Signal:
    def __init__(self, initial_power: float) -> None:
        self._signal_power: float = initial_power
        self._noise_power: float = 0
        self._latency: float = 0

    # Add latency (s)
    def add_latency(self, latency: float):
        self._latency += latency

    # Add noise power (W)
    def add_noise(self, noise: float):
        self._noise_power += noise

    def signal_power(self) -> float:
        return self._signal_power

    def latency(self) -> float:
        return self._latency

    def noise(self) -> float:
        return self._noise_power

    def snr(self) -> float:
        return self._signal_power/self._noise_power

    def copy(self) -> Self:
        n = Signal(self._signal_power)
        n._noise_power = self._noise_power
        n._latency = self._latency

        return n

class Node:
    def __init__(self, label: str, position: Point2D, transceiver: Transceiver | None) -> None:
        assert str.isalnum(label), "Node labels can only contain alphanumeric characters"
        self._label: str = label
        self._position: Point2D = position
        self._transceiver: Transceiver = transceiver if transceiver else Transceiver.FIXED_RATE

        self._edges: dict[str, 'Line'] = {}

    def add_edge(self, edge: 'Line'):
        self._edges[edge.get_end().get_label()] = edge

    def get_label(self) -> str:
        return self._label

    def get_position(self) -> Point2D:
        return self._position

    def get_edges(self) -> dict[str, 'Line']:
        return self._edges

    def set_transceiver(self, transceiver: Transceiver):
        self._transceiver = transceiver

    def transceiver(self) -> Transceiver:
        return self._transceiver

    def propagate(self, signal: Signal, path: deque[Self]) -> bool:
        n = path.popleft()
        assert n == self, "Signal should never be at this node"

        if len(path) > 0:
            return self._edges[path[0].get_label()].propagate(signal, path)

        return True

    def __repr__(self) -> str:
        return f"Node<'{self._label}'>"

class Line:
    def __init__(self, start: Node, end: Node, amplified: bool = False) -> None:
        # Length in m
        self._length: float = euc_dist(start.get_position(), end.get_position())
        self._label: str = f"{start.get_label()}{end.get_label()}"
        self._occupied: bool = False

        # Loss Coefficient in dB/km
        self._start: Node = start
        self._end: Node = end

        self._n_amplifiers: int = 0
        if amplified:
            # + 2 for pre-amp and booster
            self._n_amplifiers = 2 + self._length // 80e3

    # Duration for light to travel through the line (s)
    def latency(self) -> float:
        return self._length / C_IN_FIBER

    def ase_generation(self) -> float:
        return self._n_amplifiers * h * CBAND_CENTER * BN * db2lin(AMPLIFIER_NOISE) * (db2lin(AMPLIFIER_GAIN) - 1)

    def nli_generation(self, signal: Signal) -> float:
        n_nli = (16/(27 * pi)) * np.log((pi ** 2) * 0.5 * np.abs(self._beta_2) * (signal.symbol_rate() ** 2) * (1 / db2lin(FIBER_LOSS_COEFF))) * ((self._gamma ** 2) / (4 * db2lin(FIBER_LOSS_COEFF) * self._beta_2)) * (1 / signal.symbol_rate() ** 3)
        return (signal.signal_power() ** 3) * n_nli * (self._n_amplifiers - 1) * BN

    # Line noise (W)
    def noise(self, signal: Signal) -> float:
        if self._n_amplifiers == 0:
            return 1e-9 * signal.signal_power() * self._length
        else:
            return self.ase_generation() + self.nli_generation()

    def get_end(self) -> Node:
        return self._end

    def get_start(self) -> Node:
        return self._start

    def get_label(self) -> str:
        return self._label

    def is_occupied(self) -> bool:
        return self._occupied

    def occupy(self):
        self._occupied = True

    def unoccupy(self):
        self._occupied = False

    def __process_signal(self, signal: Signal):
        signal.add_latency(self.latency())
        signal.add_noise(self.noise(signal))

    def propagate(self, signal: Signal, path: deque[Node]) -> bool:
        n = path[0]
        assert n == self._end, "Signal should never be in this line"
        self.__process_signal(signal)
        if signal.signal_power() < 0:
            return False

        return n.propagate(signal, path)

class Connection:
    def __init__(self, start: Node, end: Node) -> None:
        self._start: Node = start
        self._end: Node = end

        self._path: deque[Node] | None = None
        self._failed: bool = False
        self._latency: float = float('inf')
        self._snr: float = 0
        self._bitrate: float = 0

    def get_start(self) -> Node:
        return self._start

    def get_end(self) -> Node:
        return self._end

    def has_path(self):
        return self._path is not None

    def set_path(self, path: list[Node]):
        self._path = deque(path)

    def failed(self) -> bool:
        return self._failed

    def set_failed(self, failed: bool):
        self._failed = failed

    def set_bitrate(self, bitrate: float):
        self._bitrate = bitrate

    def latency(self) -> float:
        return self._latency

    def snr(self) -> float:
        return self._snr

    def bitrate(self) -> float:
        return self._bitrate

    # def __deepcopy__(self, memo):
    #     cls = self.__class__
    #     res = cls.__new__(cls)
    #     memo[id(self)] = res
    #     for k, v in self.__dict__.items():
    #         setattr(res, k, deepcopy(v, memo))
    #     return res

    def __repr__(self) -> str:
        if not self._failed:
            return f"Conn<path='{self._path}' lat='{self._latency}' snr='{self._snr}' rate='{self._bitrate}'>"
        else:
            return f"Conn<from='{self._start.get_label()}' to='{self._end.get_label()}' failed'>"

class Network:
    def __init__(self, nodefile: Path) -> None:
        self._nodes: dict[str, Node] = {}
        self._lines: dict[str, Line] = {}
        # Cache weighted_paths
        self._weighted_paths: pd.DataFrame | None = None

        with open(nodefile, "r") as f:
            j: dict[str, dict] = json.load(f)

        if not isinstance(j, dict) or not all(map(lambda x: isinstance(x, str), j.keys())):
            raise Exception("Malformed network")

        # Create nodes
        for nlabel, nspec in j.items():
            n = Node(nlabel, Point2D(nspec['position'][0], nspec['position'][1]), Transceiver(nspec['transceiver']) if 'transceiver' in nspec else Transceiver.FIXED_RATE)
            self._nodes[nlabel] = n

        # Create lines
        for nlabel, nspec in j.items():
            for cn in nspec['connected_nodes']:
                l = Line(self._nodes[nlabel], self._nodes[cn])
                self._nodes[nlabel].add_edge(l)
                self._lines[l.get_label()] = l

    def nodes(self) -> Iterable[str]:
        return self._nodes.keys()

    def find_paths(self, start: str, end: str) -> list[str]:
        start_n = self._nodes[start]
        end_n = self._nodes[end]

        # BFS to construct all paths
        complete_paths: list[list[Node]] = []
        wip_paths: deque[list[Node]] = deque()
        wip_paths.append([start_n])

        while len(wip_paths) > 0:
            p = wip_paths.popleft()

            if p[-1] == end_n:
                complete_paths.append(p)

            for l in p[-1].get_edges().values():
                # No crossing over
                if l.get_end() in p:
                    continue

                np = p.copy()
                np.append(l.get_end())
                wip_paths.append(np)

        return ['->'.join(map(lambda x: x.get_label(), p)) for p in complete_paths]

    # Propagate `signal` along path
    # return the signal that would be received at the end
    def propagate(self, signal: Signal, path: str) -> Signal | None:
        result = signal.copy()
        path_d = deque([self._nodes[n] for n in path.split('->')])
        if path_d[0].propagate(result, path_d):
            return result

        return None

    def __to_nodelist(self, path: str) -> list[Node]:
        return [self._nodes[n] for n in path.split('->')]

    def __is_available(self, path: list[Node]) -> bool:
        for n1, n2 in itertools.pairwise(path):
            l = n1.get_edges()[n2.get_label()]
            if l.is_occupied():
                return False

        return True

    def __occupy_path(self, path: list[Node]):
        for n in path:
            for l in n.get_edges().values():
                l.occupy()

    def stream(self, conns: list[Connection], optimize_for: Literal['snr', 'latency'] = 'latency', occupy_paths: bool = False) -> list[Connection]:
        assert optimize_for in ['latency', 'snr'], "Unknown optimization target"

        # Clear all lines before streaming
        for l in self._lines.values():
            l.unoccupy()

        conns_new = deepcopy(conns)

        wp = self.weighted_paths(INITIAL_SIGNAL_POWER)

        for c in conns_new:
            possible_paths = wp[(wp['path'].str[0] == c.get_start().get_label()) & (wp['path'].str[-1] == c.get_end().get_label())]
            if optimize_for == 'latency':
                possible_paths.sort_values('latency', ascending=True)
            else:
                possible_paths.sort_values('snr', ascending=False)

            for p in possible_paths.itertuples():
                np = self.__to_nodelist(p[1])
                if self.__is_available(np):
                    if occupy_paths:
                        self.__occupy_path(np)

                    c.set_path(np)
                    c._latency = p[2]
                    c._snr = p[4]
                    c.set_bitrate(calculate_bit_rate(db2lin(c._snr), np[0].transceiver()))
                    break

            if not c.has_path():
                c.set_failed(True)
                continue

        return conns_new

    def weighted_paths(self, signal_power: float) -> pd.DataFrame:
        # Return the cached one if already calculated
        if self._weighted_paths is not None:
            return self._weighted_paths

        signal = Signal(signal_power)
        all_paths = map(lambda x: self.find_paths(*x), itertools.permutations(self._nodes.keys(), r=2))
        propagated_sigs = [(p, self.propagate(signal, p)) for p in itertools.chain(*all_paths)]
        df_content = [(p, s.latency(), s.noise(), lin2db(s.snr())) for p, s in propagated_sigs]
        df = pd.DataFrame.from_records(df_content, columns=['path', 'latency', 'noise', 'snr'])

        self._weighted_paths = df
        return self._weighted_paths

    def create_connection(self, start: str, end: str):
        return Connection(self._nodes[start], self._nodes[end])

    def draw(self, axes: plt.Axes):
        marker = mplm.MarkerStyle('.', fillstyle='full')
        marker_path = marker.get_path().transformed(marker.get_transform())

        node_collection = mcoll.PathCollection(
            (marker_path, ),
            sizes=[2000 for _ in self._nodes.items()],
            offsets=[n._position for n in self._nodes.values()],
            offset_transform=axes.transData,
            facecolors='white',
            edgecolors='dodgerblue',
            alpha=1
            )
        node_collection.set_transform(mtrans.IdentityTransform())
        node_collection.set_zorder(9)
        axes.add_collection(node_collection)

        for nlabel, n in self._nodes.items():
            axes.text(*n.get_position(), nlabel, zorder=10, ha='center', va='center', clip_on=True)

        # TODO: do this properly, lines are supposed to be colored the color of the starting ends transceiver
        for n1, n2 in itertools.combinations(self._nodes.values(), r=2):
            n1_lines = [l for l in n1.get_edges().values() if l.get_end() == n2]
            n2_lines = [l for l in n2.get_edges().values() if l.get_end() == n1]
            lines = n1_lines + n2_lines
            lcount = len(lines)
            s_p, e_p = (np.array(n1.get_position()), np.array(n2.get_position()))

            # center point
            c_m = (s_p + e_p)/2
            # vector from s to e
            r_or = (e_p - s_p)

            # unit vector perpendicular to r_or
            r = r_or[[1,0]]
            r[1] *= -1
            r = r/np.linalg.norm(r)

            fig = axes.get_figure()
            dpi_scale_3p = mtrans.ScaledTranslation(r[0] * 3 / 72., r[1] * 3  / 72., fig.dpi_scale_trans)
            line_trans = axes.transData
            for i in range(lcount//2 + 1):
                line_trans -= dpi_scale_3p

            for i, l in enumerate(lines):
                k = i - lcount//2
                print(i, k)
                line_trans += dpi_scale_3p
                if k == 0 and lcount % 2 == 0:
                    line_trans += dpi_scale_3p
                axes.plot([s_p[0], e_p[0]], [s_p[1], e_p[1]], transform=line_trans, linewidth=1.5, label=f"{l.get_label()}", color=TRANSCEIVER_COLORS[l.get_start().transceiver()])
                #axes.text(*c_m, f"{n1.get_label()}{n2.get_label()}", transform=line_trans, fontsize=6, va='center', ha='center', rotation=360 * np.arctan(r_or[1] / r_or[0]) / (2*pi), rotation_mode='anchor')

        axes.set_xticks([])
        axes.set_yticks([])

        axes.set_aspect('equal')
        axes.autoscale_view()

    def set_transceivers(self, transceiver: Transceiver, nodes: list[str] | None = None):
        if not nodes:
            nodes = list(self._nodes.values())
        else:
            nodes = [self._nodes[n] for n in nodes]

        for n in nodes:
            n.set_transceiver(transceiver)
