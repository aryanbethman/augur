"""augur.verify.client: a network failure costs the call it hit, not the whole real-GPU run."""

import asyncio
import csv
import json

from augur.trace.workload import Workload
from augur.verify.client import Client, main


def trace(tmp_path, n=2):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps({"input_toks": 8, "output_toks": 4, "arrival_time_ns": 0}) + "\n"
                            for _ in range(n)))
    return str(path)


async def with_server_that_hangs_up(run):
    async def hang_up(reader, writer):
        await reader.read(100)
        writer.close()

    server = await asyncio.start_server(hang_up, "127.0.0.1", 0)
    async with server:
        return await run(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")


def test_dropped_connections_are_recorded_and_the_results_still_written(tmp_path, capsys):
    out = tmp_path / "new" / "folder" / "real.csv"

    async def run(url):
        return await asyncio.to_thread(main, ["--url", url, "--model", "m", "--workload", trace(tmp_path),
                                              "--output", str(out)])

    assert asyncio.run(with_server_that_hangs_up(run)) == 1
    with open(out) as f:
        assert list(csv.DictReader(f)) == []
    assert json.loads(capsys.readouterr().out)["failed"] == 2


def test_a_timeout_fails_only_its_call(tmp_path):
    class TimedOut:
        def post(self, url, json):
            raise TimeoutError("no response")

    workload = Workload(trace(tmp_path, 1), filler_vocab=1000 + 1)
    client = Client("http://unused", "m", workload, None)
    asyncio.run(client._send(TimedOut(), next(iter(workload.entries()))))
    assert client.failed == [("req_0", "TimeoutError: no response")]
    assert client.recorder.done == []
