import atexit
import ctypes
import json
import os
from pathlib import Path

if os.environ.get("GDN_PHASE6_FULL_OPP") == "1":
    ctypes.CDLL(
        os.environ["GDN_PHASE6_FULL_OP_API"],
        mode=os.RTLD_GLOBAL | os.RTLD_NOW,
    )

if os.environ.get("GDN_GRAPH_REPLAY_FILE"):
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    original_run_fullgraph = CudaGraphManager.run_fullgraph
    replay_count = 0

    def observed_run_fullgraph(self, desc):
        global replay_count
        result = original_run_fullgraph(self, desc)
        replay_count += 1
        if replay_count == 1:
            marker = Path(f"{os.environ['GDN_GRAPH_REPLAY_FILE']}.{os.getpid()}.ready")
            marker.write_text(f"{desc}\n")
        return result

    CudaGraphManager.run_fullgraph = observed_run_fullgraph

    def write_replay_summary():
        if replay_count:
            output = Path(f"{os.environ['GDN_GRAPH_REPLAY_FILE']}.{os.getpid()}.json")
            output.write_text(json.dumps({"full_graph_replay_calls": replay_count}) + "\n")

    atexit.register(write_replay_summary)
