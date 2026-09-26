"""Render the case-study attribution overlay from the PyTorch Geometric checkpoint.

This replaces the historical overlay figure, whose atom scores came from the
earlier model and from a featurizer that emitted atom rows sorted by degree, so
its index labels did not correspond to RDKit atom indices (docs/discrepancies.md
D-011). Every label drawn here is an RDKit atom index, so a reader can check it
against the structure.

Atom indices are printed on the drawing for exactly that reason: the point of
the revision is that the labels are now verifiable.

Usage:
    python scripts/plot_case_study_attribution.py
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
import torch
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model4.explain import ExplainerSettings, atom_report, explain_molecule, top_k_atoms
from src.model4.training import TrainingConfig, load_selected_model

RDLogger.DisableLog("rdApp.*")

CASE_STUDY = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(CCCNc3c4c(nc5cc(Cl)ccc35)CCCC4)CC1)C2"
OUTPUT_DIR = REPOSITORY_ROOT / "results/figures"
DPI = 300
COLORMAP = "viridis"  # perceptually uniform, and readable in greyscale

plt.rcParams.update(
    {
        "font.size": 9,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "svg.hashsalt": "bsj-1200r2",
    }
)


def load_checkpoint(protocol: str):
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/model4_pyg.yaml").read_text(encoding="utf-8"))
    run_dir = REPOSITORY_ROOT / "results/model4" / protocol
    block = config["model"]
    training_config = TrainingConfig(
        model_kwargs=dict(
            node_feat_dim=block["node_feat_dim"],
            hidden_dims=tuple(block["hidden_dims"]),
            dense_dim=block["dense_dim"],
            dropout_rate=block["dropout_rate"],
            n_classes=block["n_classes"],
            batch_norm=block["batch_norm"],
        )
    )
    state = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    model = load_selected_model(state, training_config)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return model, config, metrics


def draw_molecule(smiles: str, scores: list[float], top_bonds: list[dict], width: int, height: int) -> bytes:
    """RDKit drawing with every atom shaded by its normalized attribution."""
    mol = Chem.MolFromSmiles(smiles)
    mapper = cm.ScalarMappable(norm=mcolors.Normalize(0.0, 1.0), cmap=COLORMAP)

    atom_colors = {}
    for index in range(mol.GetNumAtoms()):
        red, green, blue, _ = mapper.to_rgba(scores[index])
        atom_colors[index] = (red, green, blue)

    bond_indices, bond_colors = [], {}
    for bond in top_bonds:
        rd_bond = mol.GetBondBetweenAtoms(bond["begin_atom"], bond["end_atom"])
        if rd_bond is None:
            continue
        bond_indices.append(rd_bond.GetIdx())
        bond_colors[rd_bond.GetIdx()] = (0.85, 0.10, 0.10)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = True          # the labels must be checkable
    options.highlightBondWidthMultiplier = 16
    options.fillHighlights = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightBonds=bond_indices,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", default=CASE_STUDY)
    parser.add_argument("--protocol", default="reproduction")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--stem", default="figure_case_study_attribution")
    args = parser.parse_args()

    model, config, metrics = load_checkpoint(args.protocol)
    explainer_block = config["explainer"]
    settings = ExplainerSettings(
        algorithm_epochs=explainer_block["algorithm_epochs"],
        explanation_type=explainer_block["explanation_type"],
        node_mask_type=explainer_block["node_mask_type"],
        edge_mask_type=explainer_block["edge_mask_type"],
        mode=explainer_block["mode"],
        task_level=explainer_block["task_level"],
        return_type=explainer_block["return_type"],
        seed=config["reproducibility"]["seed"],
    )

    explanation = explain_molecule(model, args.smiles, settings)
    importance = explanation["atom_importance_raw"]
    table = {row["node_index"]: row for row in atom_report(args.smiles, importance)}
    scores = [table[index]["importance_normalized"] for index in range(explanation["n_atoms"])]
    top_atoms = sorted(top_k_atoms(importance, args.top_k), key=lambda i: -scores[i])
    top_bonds = sorted(explanation["bond_importance"], key=lambda b: -b["importance"])[:3]

    print(f"checkpoint: epoch {metrics['selected_epoch']} ({metrics['protocol_label']})")
    print(f"prediction: {explanation['prediction']}")
    print("\ntop atoms:")
    for index in top_atoms:
        print(f"   {table[index]['atom_label']:>6}  {scores[index]:.2f}")
    print("\ntop bonds:")
    for bond in top_bonds:
        left = table[bond["begin_atom"]]["atom_label"]
        right = table[bond["end_atom"]]["atom_label"]
        print(f"   {left}-{right:<6} {bond['importance']:.4f}  {bond['bond_type']}")

    # The molecule is wide and shallow; a matching canvas keeps the structure
    # large enough that the printed atom indices stay legible at column width.
    png = draw_molecule(args.smiles, scores, top_bonds, 1800, 750)

    figure, axis = plt.subplots(figsize=(7.2, 3.1))
    axis.imshow(plt.imread(io.BytesIO(png)))
    axis.axis("off")

    mapper = cm.ScalarMappable(norm=mcolors.Normalize(0.0, 1.0), cmap=COLORMAP)
    bar = figure.colorbar(mapper, ax=axis, fraction=0.045, pad=0.02)
    # Kept short so constrained_layout does not clip it; the caption carries the
    # full definition of the scale.
    bar.set_label("Normalized atom attribution", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    labels = ", ".join(table[index]["atom_label"] for index in top_atoms)
    axis.set_title(
        f"Top-{args.top_k} attributed atoms: {labels}\n"
        "Red bonds mark the three highest-scoring bonds",
        fontsize=8.5,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("svg", "png"):
        path = OUTPUT_DIR / f"{args.stem}.{suffix}"
        if suffix == "svg":
            figure.savefig(path, dpi=DPI, format=suffix, metadata={"Date": None})
        else:
            figure.savefig(path, dpi=DPI, format=suffix)
        written.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    plt.close(figure)
    print("\nWrote " + ", ".join(written))


if __name__ == "__main__":
    main()
