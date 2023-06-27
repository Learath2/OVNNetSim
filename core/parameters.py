import scipy.constants as con

C_IN_FIBER = (2 / 3) * con.c
INITIAL_SIGNAL_POWER = 1e-3         # 1mW
BER_THRESHOLD = 10e-3               # BER Threshold
RS = 32e9                           # Symbol rate = 32 GHz (GBaud technically)
BN = 12.5e9                         # ENB = 12.5 GHz


DEFAULT_TRX_OUTPUT_POWER = 1e-3     # 1mW
DEFAULT_TRX_NOISE = 3               # 3dB

AMPLIFIER_GAIN = 13                 # 13dB
AMPLIFIER_NOISE = 3                 # 3dB

CBAND_CENTER = 193.414e12           # 193.414 THz
FIBER_LOSS_COEFF = 1                # 1 dB/km
