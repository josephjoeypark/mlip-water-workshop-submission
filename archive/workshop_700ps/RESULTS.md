> Archived 700 ps workshop snapshot. These bulk results are superseded. [Use the final release](../../README.md). Commands below run from the repository root.

# Results and discussion

This manuscript-ready section links numerical values to the
committed source tables so that the prose, tables, and figures can be audited
together.

## Parameter-efficient molecular adaptation

We evaluated six deployed potentials formed by crossing two foundation
contexts, MACEOFF23(M) and MACELES-OFF, with three molecular-data conditions:
zero shot (Baseline), monomer--dimer adaptation (G1-G2), and monomer--hexamer adaptation
(G1-G6). Baseline denotes the released foundation checkpoint and therefore involves no
training in this study. G1-G2 and G1-G6 were adapted independently with rank-32
LoRA ($\alpha=32$) while the foundation weights were frozen. Every adapted
result reported below uses the exponential-moving-average checkpoint at exactly
123,200 optimizer updates and is averaged over training seeds 17, 29, and 43,
unless explicitly identified as the deployed seed-17 model.

The molecular dataset contains 1,000 parent structures divided by parent into
800 training, 100 validation, and 100 locked-test parents. Keeping every child
cluster from a parent in the same split prevents parent leakage. Expansion into
aggregation groups produces 5,600/700/700 G1-G2 records and
8,800/1,100/1,100 G1-G6 records; these larger record counts should not be
described as additional independent parent structures. Figure 1 uses the
parent-held-out validation split. The locked test split was not used for model
selection or for the plotted values.

**Table 1. Model capacity and LoRA adaptation budget.** The trainable fraction
is $N_{\mathrm{LoRA}}/(N_{\mathrm{base}}+N_{\mathrm{LoRA}})$. G1-G2 and G1-G6 use
the same LoRA architecture; their data and training paths differ.

| Foundation | Canonical base or G1-G6-fused parameters | MACE LoRA | LES-specific LoRA | Total LoRA | Total during LoRA training | Trainable fraction |
|---|---:|---:|---:|---:|---:|---:|
| MACEOFF23(M) | 1,428,368 | 235,392 | 0 | 235,392 | 1,663,760 | 14.15% |
| MACELES-OFF | 1,891,360 | 235,392 | 117,824 | 353,216 | 2,244,576 | 15.74% |

The MACELES adapters contain the same 235,392-parameter MACE component plus
117,824 parameters in three LES latent-readout adapter pairs. The deployed G1-G2
fused exports contain eight additional non-LoRA scalar parameters
(1,428,376 for MACEOFF23(M) / G1-G2 and 1,891,368 for MACELES-OFF / G1-G2); G1-G6 and Baseline match the
canonical counts in Table 1. Fusion removes all separate `lora_A` and `lora_B`
tensors from the deployment checkpoints. Machine-readable values are in
[`table1_model_parameters.csv`](../../model_training/data/model_parameters.csv).

## Molecular transfer across aggregation order

Figure 1 tests whether molecular adaptation transfers from the aggregation
orders represented during fitting to larger held-out clusters. The released Baseline
models have force-component RMSEs of approximately 85--101 meV $\AA^{-1}$
across G1--G6. Both adaptations reduce errors on their covered groups by more
than an order of magnitude. G1-G2 remains accurate on G1--G2 but its force error
rises sharply outside its training support, reaching
$17.29\pm0.10$ meV $\AA^{-1}$ for MACEOFF23(M) / G1-G2 and
$20.20\pm0.59$ meV $\AA^{-1}$ for MACELES-OFF / G1-G2 at G6. By contrast, G1-G6, which
includes G1--G6 during adaptation, reaches $8.78\pm0.06$ and
$8.19\pm0.06$ meV $\AA^{-1}$ at G6 for the OFF and LES foundations,
respectively. The corresponding energy trends are consistent with broader
coverage, although the MACELES-OFF / G1-G6 energy error is not uniformly lower than
MACELES-OFF / G1-G2 at every aggregation order.

These results support a coverage claim rather than a pure cluster-size causal
claim. G1-G2 and G1-G6 differ in both included groups and training implementation,
and OFF and LES are distinct foundation contexts rather than a controlled
electrostatics ablation. The open seed-17 markers also show why MD deployment
should be distinguished from the three-seed training average: seed-to-seed
variation is modest for forces but can be visible for energy, especially in
the LES foundation.

## Exploratory Born-effective-charge response

We next asked whether the LES latent polarization carries a water-like local
charge response even though BECs were not target labels in the molecular
adaptation. For one 64-water configuration, the isotropic response was
calculated for every atom as

$$
Z^*_{\mathrm{iso}}=\frac{Z^*_{xx}+Z^*_{yy}+Z^*_{zz}}{3}.
$$

The raw LES polarization has a global sign ambiguity. Table 2 applies one
global sign flip to all atoms and models solely to align the latent parity with
the conventional water sign (positive H and negative O). No Ewald
normalization, dielectric rescaling, calibration, or acoustic-sum-rule (ASR)
correction was applied.

**Table 2. Isotropic BECs for one 64-water frame compared with an RPBE-D3
liquid-water reference.** LES entries are mean $\pm$ population SD over 128 H
or 64 O atoms from the deployed seed-17 checkpoints. The reference SD describes a liquid-water ensemble, whereas the
LES SD describes atoms in a single frame, so the widths are not statistically
equivalent.

| Source/model | $Z^*_{\mathrm{iso}}$(H), $e$ | $Z^*_{\mathrm{iso}}$(O), $e$ | $2\langle Z_H^*\rangle+\langle Z_O^*\rangle$, $e$ |
|---|---:|---:|---:|
| RPBE-D3 reference | $+0.430\pm0.080$ | $-0.860\pm0.092$ | 0.000 |
| MACELES-OFF / Baseline | $+0.249\pm0.035$ | $-0.464\pm0.088$ | 0.034 |
| MACELES-OFF / G1-G2 | $+0.199\pm0.033$ | $-0.371\pm0.062$ | 0.027 |
| MACELES-OFF / G1-G6 | $+0.241\pm0.024$ | $-0.444\pm0.069$ | 0.038 |

The three LES models reproduce the expected sign pattern after parity alignment
but underestimate the RPBE-D3 response magnitude. G1-G2 reduces the magnitude
most, while G1-G6 partially restores the Baseline scale. The nonzero scalar sum and the
full tensor residuals show that the raw model output does not exactly satisfy
the ASR. RPBE-D3 is an external liquid-water benchmark, not a matched
$\omega$B97M-V/def2-TZVPD response target. Consequently, this analysis is
evidence of a latent response signal,
not a calibrated prediction of physical partial charges. A defensible BEC
claim requires multiple independent liquid frames, a prespecified ASR
treatment, and direct comparison with matched electronic-structure BEC labels.
The reference values are from
[Kretschmer et al.](https://arxiv.org/abs/2607.22903); machine-readable values
and the unmodified tensors are in
[`table2_bec_comparison.csv`](../../model_training/data/bec_frame0/bec_comparison.csv)
and [`les_bec_64water_frame0.csv`](../../model_training/data/bec_frame0/les_bec_64water_frame0.csv).

## Transfer to equilibrium density and local structure

Figure 2 moves from molecular validation to bulk equilibrium behavior. Each
state comprises three paired replicas of a flexible 64-water box propagated
for 1 ns in the isotropic NPT ensemble at 1.01325 bar. Densities and RDFs use
only 500--1,000 ps. At 295 K, the two Baseline foundations substantially overpredict
density: $1.1886\pm0.0005$ g cm$^{-3}$ for MACEOFF23(M) / Baseline and
$1.1673\pm0.0016$ g cm$^{-3}$ for MACELES-OFF / Baseline, compared with the IAPWS value
[0.9978 g cm$^{-3}$](https://www.iapws.org/relguide/LiquidWater.pdf). G1-G2 reduces but does not remove this bias. G1-G6 crosses the
reference in a foundation-dependent manner: MACEOFF23(M) / G1-G6 underpredicts it by 7.10%,
whereas MACELES-OFF / G1-G6 reaches $0.9830\pm0.0010$ g cm$^{-3}$, a -1.48% error.
The same qualitative ordering persists over 275--336 K.

Against the [298 K Soper reference](https://doi.org/10.1155/2013/279463), the first O--O maximum is approximately
$g_{OO}=2.50$ at 2.79 $\AA$. The simulations place it between 2.81 and
2.89 $\AA$. G1-G6 sharpens the peak to 2.87 for MACEOFF23(M) / G1-G6 and 2.74 for MACELES-OFF / G1-G6,
indicating stronger first-shell order than experiment. The intermolecular O--H
first maximum likewise increases to 1.36 and 1.29 for MACEOFF23(M) / G1-G6 and MACELES-OFF / G1-G6,
respectively, compared with approximately 1.04 in the 298 K Soper reference.
G1-G2 gives peak heights closer to the experimental curves but retains a large
density error. Thus, better molecular force transfer does not guarantee
simultaneous agreement in density and local structure; the bulk observables
probe the accumulated balance of short- and long-range interactions.

The 3 K mismatch between the simulated 295 K intermolecular O--H RDF and the
298 K reference should remain visible in the caption. Error bars on density
are replica SD, and RDF bands are replica SD rather than uncertainty over time
origins.

## Dynamics and collective relaxation

For transport, each selected NPT state initializes a 700 ps fixed-volume
Nose--Hoover trajectory with a 0.5 fs integration step and native 0.5 fs
oxygen-velocity and five-channel traceless-stress records. Figure 3 uses the
complete 700 ps trajectory across all six models and four temperatures. VACF diffusion and Green--Kubo viscosity are reported as the
mean running integral over 5--10 ps; state uncertainties are SEM across the
three paired replicas. Corrected diffusion uses the lane-wise Yeh--Hummer term,

$$
D_\infty=D_L+\frac{k_\mathrm{B}T\,2.837297}{6\pi\eta L},
$$

so corrected diffusion and viscosity are statistically coupled.

Across the four temperatures, the OFF adaptations provide the most balanced
agreement with experiment: mean absolute relative errors are 16.2% in
corrected diffusion and 13.7% in viscosity for MACEOFF23(M) / G1-G2, and 14.0% and 11.9%
for MACEOFF23(M) / G1-G6. The LES conditions expose a stronger trade-off. MACELES-OFF / G1-G2 lowers
viscosity but overestimates corrected diffusion, whereas MACELES-OFF / G1-G6 gives a much
closer diffusion trend while strongly overestimating viscosity. At 295 K,
MACELES-OFF / G1-G6 has $D_\infty=1.805\pm0.017$ nm$^2$ ns$^{-1}$ versus the
temperature-matched experimental 2.123, but
$\eta=2.318\pm0.043$ mPa s versus 0.958 mPa s. MACEOFF23(M) / G1-G6 at the same state gives
$D_\infty=2.357\pm0.008$ nm$^2$ ns$^{-1}$ and
$\eta=0.958\pm0.043$ mPa s.

The experimental diffusion curve uses the
[Holz--Heil--Sacco fit](https://doi.org/10.1039/B005319H), and the viscosity
curve uses the [IAPWS 0.1 MPa correlation](https://www.iapws.org/relguide/LiquidWater.pdf).

The MACEOFF23(M) / G1-G6/MACELES-OFF / G1-G6 stress decomposition helps explain this contrast. At
295 K, the LES/OFF ratios are 2.42 for the finite-window viscosity, 1.21 for
the instantaneous shear modulus, and 2.00 for the mean stress-relaxation time.
The higher MACELES-OFF / G1-G6 viscosity is therefore dominated by a longer-lived stress
correlation rather than only a larger zero-time amplitude. This is a
finite-window interpretation: three replicas and a 64-water box do not
establish thermodynamic-limit Green--Kubo convergence.

## Cross-scale interpretation and limitations

Taken together, the figures show that molecular validation, equilibrium
structure, and transport rank models differently. G1-G6 improves high-order
cluster forces for both foundations, MACELES-OFF / G1-G6 gives the best density, and
MACEOFF23(M) / G1-G6 provides the most balanced diffusion--viscosity agreement. No single
molecular RMSE is therefore a sufficient deployment criterion. The central
result is the value of a staged audit: molecular extrapolation detects coverage
failure, equilibrium simulations reveal accumulated interaction bias, and
transport resolves collective relaxation errors that are not apparent from
density alone.

The conclusions remain bounded by four limitations. First, OFF and LES are
different foundation checkpoints, so their difference is not an isolated LES
ablation. Second, Figure 1 reports validation rather than the locked test set.
Third, Table 2 is a one-frame, uncalibrated response diagnostic with an ASR
residual. Fourth, 64 waters, three replicas, and a common 700 ps NVT analysis
prefix are appropriate for screening but not a definitive size- and
time-converged transport benchmark. A final archival claim should add a
shortlisted $\geq512$-water size check, longer or additional stress replicas,
and a multi-frame electronic-response benchmark.
