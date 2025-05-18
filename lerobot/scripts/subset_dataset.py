#!/usr/bin/env python

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.common.datasets.utils import (
    write_episode,
    write_episode_stats,
    write_info,
    write_stats,
)


@dataclass
class SubsetDatasetConfig:
    src_repo_id: str
    dst_repo_id: str
    keep_first: int
    root: Path | None = None
    push_to_hub: bool = True
    private: bool = False
    branch: str | None = None


@draccus.wrap()
def main(cfg: SubsetDatasetConfig):
    selected_eps = list(range(cfg.keep_first))

    src_dataset = LeRobotDataset(cfg.src_repo_id, episodes=selected_eps, root=cfg.root)

    dst_root = src_dataset.root.parent / cfg.dst_repo_id
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_dataset.root, dst_root)

    meta = LeRobotDatasetMetadata(cfg.src_repo_id, root=src_dataset.root)
    meta.repo_id = cfg.dst_repo_id
    meta.root = dst_root

    meta.info["total_episodes"] = cfg.keep_first
    meta.info["total_frames"] = len(src_dataset)
    meta.info["total_videos"] = len(meta.video_keys) * cfg.keep_first
    meta.info["total_chunks"] = selected_eps[-1] // meta.info["chunks_size"] + 1
    meta.info["splits"] = {"train": f"0:{cfg.keep_first}"}

    meta.episodes = {ep: meta.episodes[ep] for ep in selected_eps}
    meta.episodes_stats = {ep: meta.episodes_stats[ep] for ep in selected_eps}
    meta.stats = aggregate_stats(list(meta.episodes_stats.values()))

    write_info(meta.info, dst_root)
    (dst_root / "meta" / "episodes.jsonl").unlink(missing_ok=True)
    (dst_root / "meta" / "episodes_stats.jsonl").unlink(missing_ok=True)
    (dst_root / "meta" / "stats.json").unlink(missing_ok=True)
    for ep in selected_eps:
        write_episode(meta.episodes[ep], dst_root)
        write_episode_stats(ep, meta.episodes_stats[ep], dst_root)
    write_stats(meta.stats, dst_root)

    subset_dataset = LeRobotDataset(cfg.dst_repo_id, root=dst_root)
    if cfg.push_to_hub:
        subset_dataset.push_to_hub(branch=cfg.branch, private=cfg.private)


if __name__ == "__main__":
    main()
