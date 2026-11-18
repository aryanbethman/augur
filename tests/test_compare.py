"""augur.verify.compare pairs the same call in both files, whatever order each simulator released calls in."""

from augur.verify.compare import align, summarize


def row(rid, ttft, session=None, index=0, arrival=0, n_in=100, n_out=10):
    r = {"request id": str(rid), "input": str(n_in), "output": str(n_out), "arrival": str(arrival), "TTFT": str(ttft),
         "TPOT": "1", "latency": "2", "end_time": "3", "queuing_delay": "0"}
    if session is not None:
        r.update(session_id=session, sub_request_index=str(index))
    return r


def test_session_calls_pair_by_session_and_index_not_by_release_order():
    # Same lengths, released in the opposite order: ids and lengths alone would pair X#3 with Y#7.
    a = {0: row(0, 111, "X", 3), 1: row(1, 222, "Y", 7)}
    b = {0: row(0, 222, "Y", 7), 1: row(1, 111, "X", 3)}
    s = summarize(a, b)
    assert s["matched"] == 2 and s["TTFT"]["exact"] == 2


def test_flat_requests_pair_by_arrival_order():
    a = {0: row(0, 5, "", arrival=10), 1: row(1, 6, "", arrival=20), 2: row(2, 7, "S", 0, arrival=15)}
    b = {0: row(0, 7, "S", 0, arrival=15), 1: row(1, 5, "", arrival=10), 2: row(2, 6, "", arrival=20)}
    assert summarize(a, b)["TTFT"]["exact"] == 3


def test_a_call_missing_from_one_file_is_reported():
    s = summarize({0: row(0, 1, "X", 0), 1: row(1, 2, "X", 1)}, {0: row(0, 1, "X", 0)})
    assert (s["matched"], s["only_a"], s["only_b"]) == (1, 1, 0)


def test_files_without_session_columns_keep_the_length_pairing():
    a = {0: row(0, 1, n_in=100), 1: row(1, 2, n_in=200, arrival=1)}
    b = {0: row(0, 2, n_in=200, arrival=1), 1: row(1, 1, n_in=100)}
    na, nb = align(a, b)
    assert sorted((na[k]["input"], nb[k]["input"]) for k in na) == [("100", "100"), ("200", "200")]
