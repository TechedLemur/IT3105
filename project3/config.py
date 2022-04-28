from dataclasses import dataclass


class Config:
    @dataclass
    class SimWorldConfig:
        L1 = 1
        L2 = 1
        m1 = 1
        m2 = 1
        Lc1 = 1 / 2
        Lc2 = 1 / 2
        g = 9.81
        dt = 0.05
        N = 100
        n = 4

    class TileEncodingConfig:
        buckets = 4
        tiles = 4
