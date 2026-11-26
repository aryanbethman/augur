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


@pytest.mark.parametrize("vocab", [150000, None])
@pytest.mark.parametrize("count", [1, 127, 128, 129, 700])
def test_lazy_filler_reads_as_the_list(vocab, count):
    eager, lazy = _Filler(vocab), _Filler(vocab)
    want, got = eager.ids(count, "c#1/out"), lazy.lazy(count, "c#1/out")
    assert len(got) == count and [got[i] for i in range(count)] == want
    assert list(got[count // 2:count // 2 + 5]) == want[count // 2:count // 2 + 5] and got[-1] == want[-1]
    assert [got[i] for i in range(count - 1, -1, -1)] == want[::-1]  # out of order: windows are rebuilt
