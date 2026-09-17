"""Frozen clip and model constants."""

VOCAB = 1024
T = 8  # context frames
S = 128  # tokens per frame
FRAMES = 1200
H = 128
W = 256
FPS = 20
SYMBOLS_PER_CLIP = FRAMES * S
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
FFN_DIM = 768
FRAME_ROWS = 8
FRAME_COLS = 16
BIT_DEPTH = 10
RAW_BPP = BIT_DEPTH  # packed 10-bit symbols vs raw storage reference
