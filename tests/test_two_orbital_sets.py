# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Physics/algebra validation for the genuine two-orbital-set (UHF-shaped) path.

See docs/uhf_two_orbital_sets_plan.md for the plan this implements (steps 1-3
only: spin-resolved state in run_low_level, EmbeddedOrbitals' second orbital
set, and build_orbitals(two_orbital_sets=True)). Steps 1-3 are NOT implemented
yet, so the tests exercising them are removed for now -- only the
already-existing run_low_level() open-shell guard is tested here. Re-add the
step 1-3 tests once that work lands.

Runs against a *real* ``embasi.embedding.ProjectionEmbedding`` (no mock), an
OH radical (doublet, one unpaired electron) embedded in a water environment
at PBE0-in-PBE/sto-3g -- the same system used to validate EmbASI's own
open-shell support this session. Marked ``embasi``/``slow``, skipped unless
``EMBASI_AVAILABLE=1``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.embasi, pytest.mark.slow]


def test_run_low_level_refuses_open_shell_on_restricted_adapter():
    """A restricted adapter (unrestricted=False) must fail loudly on a
    genuinely open-shell EmbASI reference, not silently drop the beta
    channel -- this is a pre-existing guard in _as_ao_matrix, exercised here
    because a real open-shell run is exactly what step 1-3's work newly
    makes reachable."""
    import pyscf
    from ase import Atoms
    from embasi.embedding import ProjectionEmbedding
    from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
        PySCFIntegrals,
    )

    atoms = Atoms("OHOHH", positions=[[0, 0, 0], [0, 0, 0.97], [4, 0, 0], [4, 0, 0.96], [4.9, 0, -0.3]])
    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="sto-3g", spin=1)
    mf_ll, mf_hl = mol.UKS(xc="PBE"), mol.UKS(xc="PBE0")
    projection = ProjectionEmbedding(
        atoms, embed_mask=[1, 1, 2, 2, 2], calc_base_ll=PySCF(method=mf_ll),
        calc_base_hl=PySCF(method=mf_hl), projection="level-shift", mu_val=1.0e6, parallel=False,
    )
    restricted_adapter = ProjectionEmbeddingAdapter(
        projection, PySCFIntegrals(mf_hl, mf_ll), mu=1.0e6, unrestricted=False
    )
    with pytest.raises(NotImplementedError):
        restricted_adapter.run_low_level()
