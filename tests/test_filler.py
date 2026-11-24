"""Filler token ids (augur.trace.workload): in-vocabulary ids must stay exactly what random.randrange gave, because
real runs were driven with those prompts (augur.verify.client) and augur rebuilds the same ones."""

import random
import zlib

import pytest

from augur.trace.workload import _Filler, _randranges


@pytest.mark.parametrize("lo,hi", [(1000, 150000), (1000, 1001), (0, 1 << 32), (5, 7)])
@pytest.mark.parametrize("count", [0, 1, 3, 1000, 24000])
def test_bulk_draw_gives_randranges_ids(lo, hi, count):
    want = random.Random(count + hi)
    got = random.Random(count + hi)
    assert _randranges(got, lo, hi, count) == [want.randrange(lo, hi) for _ in range(count)]


def test_vocab_filler_is_seeded_by_the_key():
    rng = random.Random(zlib.crc32(b"session_3#2/in"))
    assert _Filler(150000).ids(500, "session_3#2/in") == [rng.randrange(1000, 150000) for _ in range(500)]
