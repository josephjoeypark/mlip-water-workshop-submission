# 64-water initial configuration

`liquid-64.xyz` is the exact file from Bingqing Cheng's public `cacefit`
repository at commit `c3ae70066ec86f5f6a51a7146c895dfd0661a013`:

- configuration: <https://github.com/BingqingCheng/cacefit/blob/c3ae70066ec86f5f6a51a7146c895dfd0661a013/md-water/liquid-64.xyz>
- original MD example: <https://github.com/BingqingCheng/cacefit/blob/c3ae70066ec86f5f6a51a7146c895dfd0661a013/md-water/md.py>
- upstream repository: <https://github.com/BingqingCheng/cacefit>

The file contains 64 flexible water molecules (192 atoms) in a periodic cubic
cell with edge length 12.415238 A. Its SHA-256 is:

```text
347fcf41584ca834d1bdb1acc622a9ad9329091ec1d033e423dc29c6ef03ab69
```

The NPT protocol descends from Bingqing Cheng's LES water example at commit
`a886785caa4182a0effb95d95f0e402881adfc4d`:

- upstream NPT code: <https://github.com/ChengUCB/les_fit/blob/a886785caa4182a0effb95d95f0e402881adfc4d/MD_Simulations/Water/NPT/NPT-ase.py>

That example uses BFGS to 0.2 eV/A, 5 ps of thermostat-only ASE NPT, and
300,000 IsotropicMTKNPT steps (300 ps). This study preserves that lineage but
extends production to 1,000,000 steps (1 ns), evaluates four temperatures and
three replicas, and adds restart/integrity checks. `run_nvt.py` then performs
700 ps of fixed-volume NVT at a 0.5 fs step from the representative NPT frame.

This citation describes provenance; it does not imply endorsement of the
adapted protocol by the upstream author.
