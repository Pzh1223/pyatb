import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


if "mpi4py" not in sys.modules:
    class _DummyComm:
        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def Barrier(self):
            return None

        def reduce(self, value, root=0, op=None):
            return value

    class _DummyOp:
        @staticmethod
        def Create(func, commute=False):
            return func

    dummy_mpi = types.SimpleNamespace(COMM_WORLD=_DummyComm(), Op=_DummyOp, SUM="sum")
    sys.modules["mpi4py"] = types.SimpleNamespace(MPI=dummy_mpi)


if "pyatb.interface_python" not in sys.modules:
    class _DummyInterfacePython:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["pyatb.interface_python"] = types.SimpleNamespace(
        interface_python=_DummyInterfacePython
    )
