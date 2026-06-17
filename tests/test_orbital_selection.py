from pathlib import Path
import shutil

from pyatb.fermi.orbital_selection import (
    build_orbital_map,
    parse_global_orbital_indices,
    resolve_atom_orbitals,
)


def test_resolves_abacus_atom_shells_to_global_nao_indices():
    root = Path(__file__).resolve().parents[1]
    stru = root / "examples" / "Si2" / "pyatb" / "STRU"

    orbital_map = build_orbital_map(stru)

    assert orbital_map.total_orbitals == 26
    assert resolve_atom_orbitals(orbital_map, 1, "all").indices == list(range(0, 13))
    assert resolve_atom_orbitals(orbital_map, 2, "all").indices == list(range(13, 26))
    assert resolve_atom_orbitals(orbital_map, 1, "2s").indices == [0, 1]
    assert resolve_atom_orbitals(orbital_map, 1, "2p").indices == list(range(2, 8))
    assert resolve_atom_orbitals(orbital_map, 1, "1d").indices == list(range(8, 13))


def test_parse_global_orbital_indices():
    assert parse_global_orbital_indices("0, 2,5") == [0, 2, 5]


def test_build_orbital_map_uses_relative_abacus_orbital_dir(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source_dir = root / "examples" / "Si2" / "pyatb"
    stru = tmp_path / "STRU"
    orbital_dir = tmp_path / "basis"
    orbital_dir.mkdir()
    shutil.copy(source_dir / "STRU", stru)
    shutil.copy(source_dir / "Si_lda_8.0au_50Ry_2s2p1d", orbital_dir / "Si_lda_8.0au_50Ry_2s2p1d")
    input_file = tmp_path / "Input"
    input_file.write_text("INPUT_PARAMETERS\n{\n    orbital_dir basis\n}\n")

    orbital_map = build_orbital_map(stru, input_file=input_file)

    assert orbital_map.total_orbitals == 26
    assert all(str(atom.orbital_file).startswith(str(orbital_dir)) for atom in orbital_map.atoms)
