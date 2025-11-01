# This is the manual (prototype)

The [`loop.py`](acatal/loop.py) is all you need to run a single active learning loop.



### Before the first training:
- Config VASP:

  - Load compiler: `module load intel`.
  - Modify `vasp_run.sh` under `os.path.dirname(acatal.__file__)/bin` to fit you own supercomputer (cluster/workstation). This file is used to run VASP calculations. For example (Linux OS), for example:


  ```shell
  #/bin/bash
  mpirun vasp_std
  ```

  - Prepare POTCAR files

  - Prepare an initial database.

    - A file contains some chemical formulas, for example the `formula.txt` file:


    ```
    Ni0.10Co0.10Fe0.10Pd0.10Pt0.60
    Ni0.10Co0.10Fe0.10Pd0.20Pt0.50
    Ni0.10Co0.10Fe0.10Pd0.30Pt0.40
    Ni0.10Co0.10Fe0.10Pd0.40Pt0.30
    Ni0.10Co0.10Fe0.10Pd0.50Pt0.20
    Ni0.10Co0.10Fe0.10Pd0.60Pt0.10
    Ni0.10Co0.10Fe0.10Pd0.70Pt0.00
    Ni0.13Co0.13Fe0.13Pd0.10Pt0.50
    Ni0.13Co0.13Fe0.13Pd0.20Pt0.40
    Ni0.13Co0.13Fe0.13Pd0.30Pt0.30
    Ni0.13Co0.13Fe0.13Pd0.40Pt0.20
    Ni0.13Co0.13Fe0.13Pd0.50Pt0.10
    Ni0.13Co0.13Fe0.13Pd0.60Pt0.00
    Ni0.16Co0.16Fe0.16Pd0.10Pt0.40
    Ni0.16Co0.16Fe0.16Pd0.20Pt0.30
    Ni0.16Co0.16Fe0.16Pd0.30Pt0.20
    Ni0.16Co0.16Fe0.16Pd0.40Pt0.10
    Ni0.16Co0.16Fe0.16Pd0.50Pt0.00
    Ni0.20Co0.20Fe0.20Pd0.10Pt0.30
    Ni0.20Co0.20Fe0.20Pd0.20Pt0.20
    Ni0.20Co0.20Fe0.20Pd0.30Pt0.10
    Ni0.20Co0.20Fe0.20Pd0.40Pt0.00
    Ni0.23Co0.23Fe0.23Pd0.10Pt0.20
    Ni0.23Co0.23Fe0.23Pd0.20Pt0.10
    Ni0.23Co0.23Fe0.23Pd0.30Pt0.00
    Ni0.26Co0.26Fe0.26Pd0.10Pt0.10
    Ni0.26Co0.26Fe0.26Pd0.20Pt0.00
    Ni0.30Co0.30Fe0.30Pd0.10Pt0.00
    ```

    - Run high-throughput DFT calculations:

    ```python
    import agat
    import acatal
    from agat.app.cata import HtDftAds
    import os
    import numpy as np
    
    HA = HtDftAds(calculation_index=0, sites='bridge', include_bulk_aimd=False,
                  include_surface_aimd=False, include_adsorption_aimd=False,
                  random_samples=1, vasp_bash_path=os.path.join(os.path.dirname(acatal.__file__), 'bin', 'vasp_run.sh'))
    
    formula = np.loadtxt('formula.txt')
    HA.run(formula)
    ```

    




Not Used.
| | |
V V V

- Modify `run.sh` under `os.path.dirname(acatal.__file__)/bin`. This file is used to initialize VASP environment and run the high-throughput DFT calculations. For example (my Linux account):

  **Example:** 
  ```shell
  . ~/intel/oneapi/setvars.sh
  python aug_data.py
  ```

^ ^ ^
| | |
Not Used.

### Step-by-step instruction:

