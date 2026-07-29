"""Optional small-sample differentiable-FEM fine-tuning for a frozen MeshGNN."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.sim_lung_graph import SimLungGraphDataset  # noqa: E402
from experiments.train_lung_mesh_gnn import (  # noqa: E402
    build_model,
    material_loss,
    move_graph,
)
from physics.fem import solve_nh_heterogeneous  # noqa: E402


def physics_loss(
    model: torch.nn.Module,
    graph: dict,
    device: torch.device,
    load_limit: int,
) -> torch.Tensor:
    train_graph = move_graph(graph, device)
    output = model([train_graph])
    node_log_E = output["node_log_E"]
    assert isinstance(node_log_E, list)
    E_nodes = node_log_E[0].exp().to(torch.float64)
    physics = graph["physics"]
    nodes = physics["nodes"].to(device)
    elems = physics["elems"].to(device)
    fixed = physics["fixed"].to(device)
    surface_ids = physics["surface_node_ids"].to(device)
    nu = physics["nu"].to(device)
    observations = physics["surface_observations"]
    if observations is None:
        raise ValueError("Physics fine-tuning requires 3-D surface observations")
    losses = []
    for force, target in zip(
        physics["forces"][:load_limit], observations[:load_limit]
    ):
        displacement = solve_nh_heterogeneous(
            nodes,
            elems,
            E_nodes,
            nu,
            force.to(device),
            fixed,
        )
        prediction = displacement.view(-1, 3)[surface_ids]
        target = target.to(device)
        scale = target.square().mean().sqrt().clamp_min(1e-6)
        losses.append(((prediction - target) / scale).square().mean())
    supervised, _ = material_loss(output, [train_graph])
    return torch.stack(losses).mean() + 0.05 * supervised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-patients", type=int, default=1)
    parser.add_argument("--load-limit", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    model = build_model(
        config["model"],
        config["input_dim"],
        config["hidden_dim"],
        config["layers"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(state["model"])
    dataset = SimLungGraphDataset(
        args.dataset,
        split="train",
        observation_key="noisy",
        experiments_limit=config.get("experiments_limit"),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        losses = []
        model.train()
        for index in range(min(args.max_patients, len(dataset))):
            optimizer.zero_grad(set_to_none=True)
            loss = physics_loss(model, dataset[index], device, args.load_limit)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "physics_loss": sum(losses) / len(losses)})
        print(json.dumps(history[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": state["epoch"],
            "config": {**config, "physics_fine_tuned": True},
        },
        args.out,
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "evidence_scope": "small-sample differentiable FEM fine-tuning",
                "patient_count": min(args.max_patients, len(dataset)),
                "load_limit": args.load_limit,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
