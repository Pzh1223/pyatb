from pyatb.io.default_input import INPUT, function_switch, need_rR_matrix
from pyatb.io.input import get_general_parameter


def test_cohp_input_block_is_registered_without_rr_requirement():
    assert "COHP" in function_switch
    assert "COHP" in INPUT
    assert "COHP" not in need_rR_matrix

    cohp = INPUT["COHP"]
    assert cohp["stru_file"][-1] is None
    assert cohp["method"][-1] == "COHP"
    assert cohp["spin"][-1] == "sum"
    assert cohp["kpoint_mode"][-1] is None


def test_variable_length_orbital_selector_stops_at_next_known_parameter():
    data = ["atom_i_orbs", "2s", "2p", "atom_j_orbs", "all", "de", "0.02"]
    known = {"atom_i_orbs", "atom_j_orbs", "de"}

    assert get_general_parameter("atom_i_orbs", [str, -1, "all"], data, known) == "2s,2p"
    assert get_general_parameter("atom_j_orbs", [str, -1, "all"], data, known) == "all"
    assert get_general_parameter("de", [float, 1, 0.05], data, known) == 0.02
