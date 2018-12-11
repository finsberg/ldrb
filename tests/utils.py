import pytest
skip_in_parallel = pytest.mark.skipif(MPI.size(MPI.comm_world) > 1,
                                      reason="This test should only be run in serial.")