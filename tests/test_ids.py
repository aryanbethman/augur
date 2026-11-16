from augur.trace.workload import call_id


def test_augur_client_ids():
    # augur.verify.client sends "augur.<call>." and real vLLM wraps it.
    assert call_id("cmpl-augur.req_7.-0") == "req_7"
    assert call_id("cmpl-augur.sess_3#2.-0-a1b2c3") == "sess_3#2"


def test_vllm_bench_serve_ids():
    # `vllm bench serve` request i is augur's req_<i>.
    assert call_id("cmpl-bench-e39b9ac0-41-0-9f8e7d6c") == "req_41"
    assert call_id("bench-e39b9ac0-0-0") == "req_0"


def test_augur_internal_ids():
    assert call_id("req_5") == "req_5"
    assert call_id("req_5@p") == "req_5"  # PD prefill half
    assert call_id("req_5@d") == "req_5"  # PD decode half
