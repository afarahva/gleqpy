"""
Regenerate the reference ("golden") values for the gleqpy test suite.

Run this ONCE, in the reference environment (ase_2025), whenever an intended
change to the numerics means the saved references should be updated:

    python tests/generate_goldens.py

It writes tests/data/*.npz.  Review the git diff before committing.
"""

import numpy as np
import _helpers as H


def _save(name, d):
    path = H._data_path(name)
    np.savez(path, **d)
    print(f"wrote {path}  ({', '.join(f'{k}{np.asarray(v).shape}' for k, v in d.items())})")


def main():
    import os
    os.makedirs(H.DATA_DIR, exist_ok=True)

    # 1) in-house MD golden trajectory
    _save("md_inhouse.npz", H.inhouse_md_run())

    # 2) ASE MD golden trajectory
    _save("md_ase.npz", H.ase_md_run())

    # 3) precomputed arrays + expected memory kernels
    arrays = H.make_memory_input_arrays()
    kernels = H.compute_memory_kernels(arrays)
    _save("memory_arrays.npz", {**arrays, **{f"expected_{k}": v for k, v in kernels.items()}})

    # 5) projection-operator golden
    _save("projection.npz", H.projection_case())

    print("done.")


if __name__ == "__main__":
    main()
