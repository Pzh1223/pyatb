# COHP

The COHP function evaluates atom-pair crystal orbital Hamilton population
spectra from the ABACUS Hamiltonian matrix, overlap matrix and NAO
eigenvectors already used by PyATB.

The implemented value for one band and one k point is

```text
sum_ij Re[ C_i,n(k)^* M_ij(k) C_j,n(k) ]
```

where `M` is `H(k)` for COHP and `S(k)` for COOP. The selected orbital sets
belong to two atom indices in the ABACUS `STRU` file. The spectrum is broadened
on an energy grid with the same Gaussian smearing convention used by the PyATB
PDOS/JDOS modules.

## Input block

```text
COHP
{
    stru_file          STRU
    atom_i_index      1
    atom_j_index      2
    atom_i_orbs       all
    atom_j_orbs       2p
    method            COHP
    spin              sum
    e_range           -8.0 8.0
    de                0.02
    sigma             0.08
    invert            1
    shift_to_efermi   1
    output_prefix     COHP
    kpoint_mode       mp
    mp_grid           8 8 8
}
```

`atom_i_orbs` and `atom_j_orbs` accept `all`, shell selectors such as `s`,
`p`, `d`, `2p`, or comma/space separated global orbital indices. Atom indices
are 1-based and follow the atom order in `STRU`.

## Output

The output directory is `Out/COHP`. The main spectrum is written to
`COHP.dat` by default, with columns `energy_eV` and `COHP`. If
`shift_to_efermi` is set to `1`, the energy axis is `E - E_F`.

`COHP.meta.json` records the selected atoms, selected global orbital indices,
orbital shell mapping and output file names. `plot_cohp.py` is generated in the
same directory.

## Notes

COHP currently supports `nspin = 1` and collinear `nspin = 2`. For spin-polarized
data, set `spin` to `sum`, `up`, or `down`.
