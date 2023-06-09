from enum import StrEnum
from utility import Point2D, euc_dist
from parameters import C_IN_FIBER

class Transceiver(StrEnum):
    FIXED_RATE: 'fixed-rate'
    FLEX_RATE: 'flex-rate'
    SHANNON: 'shannon'

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

class Node:
    def __init__(self, label: str, position: Point2D, transceiver: Transceiver | None) -> None:
        assert str.isalnum(label), "Node labels can only contain alphanumeric characters"
        self._label: str = label
        self._position: Point2D = position
        self._transceiver: Transceiver = transceiver if transceiver else Transceiver.FIXED_RATE

        self._edges: list[Line] = []

    # Latency added by processing on the Node (s)
    def latency(self) -> float:
        return 0

    # Noise added by processing on the Node (W)
    def noise(self) -> float:
        return 0

    def get_label(self) -> str:
        return self._label

    def get_position(self) -> Point2D:
        return self._position

    def process_signal(self, signal: Signal):
        signal.add_latency(self.latency())
        signal.add_noise(self.noise())


class Line:
    def __init__(self, start: Node, end: Node) -> None:
        self._length = euc_dist(start.get_position(), end.get_position())
        self._label = f"{start.get_label()}{end.get_label()}"

    # Duration for light to travel through the line (s)
    def latency(self) -> float:
        return self._length / C_IN_FIBER

    # Line noise (W)
    def noise(self) -> float:
        pass

    def process_signal(self, signal: Signal):
        signal.add_latency(self.latency())
        signal.add_noise()

class Connection:
    def __init__(self, path: list[Node]) -> None:
        self._path = path

class Network:
    def __init__(self) -> None:
        self._nodes: dict[str, Node]
