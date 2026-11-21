"""augur.verify.stepcheck rebuilds a speculative run's steps: a request past its prompt that was scheduled 1 + drafts
tokens is a decode, and only the tokens it emitted (accepted drafts + 1) enter its KV cache."""

from augur.verify.stepcheck import real_steps


def test_a_verification_step_is_a_decode_and_rejected_drafts_do_not_advance_the_context():
    ev = [
        {"ev": "add", "t": 0, "rid": "a", "prompt": 10},
        {"ev": "sched", "t": 1, "k": 1, "tok": {"a": 10}, "new": {"a": 0}, "pre": []},
        {"ev": "upd", "t": 2, "k": 1, "out": {"a": 1}, "fin": [], "left": 1},
        {"ev": "sched", "t": 3, "k": 2, "tok": {"a": 4}, "new": {}, "pre": []},   # 1 + 3 drafts, 1 accepted: 2 emitted
        {"ev": "upd", "t": 4, "k": 2, "out": {"a": 2}, "fin": [], "left": 1},
        {"ev": "sched", "t": 5, "k": 3, "tok": {"a": 4}, "new": {}, "pre": []},
        {"ev": "upd", "t": 6, "k": 3, "out": {"a": 4}, "fin": [], "left": 1},
    ]
    shapes = [s for s, _ in real_steps(ev)]
    prefill, v1, v2 = (sh.seqs[0] for sh in shapes)
    assert not prefill.is_decode and prefill.num_new == 10
    assert v1.is_decode and v1.num_new == 4 and v1.num_computed_before == 10
    assert v2.is_decode and v2.num_computed_before == 12  # 10 + the 2 tokens emitted, not the 4 scheduled
