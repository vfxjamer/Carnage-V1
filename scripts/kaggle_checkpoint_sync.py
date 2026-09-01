#!/usr/bin/env python3
"""Restore or coalescing-publish Carnage checkpoints with a private Kaggle dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

import kagglehub

REQUIRED_CHECKPOINT_FILES = {
    "POLICY.lt", "CRITIC.lt", "POLICY_OPTIM.lt", "CRITIC_OPTIM.lt",
    "RUNNING_STATS.json", "CARNAGE_METADATA.json",
}


def checkpoint_is_valid(path: Path) -> bool:
    """Mirror the inexpensive portion of the trainer's checkpoint admission check."""
    if not path.is_dir() or not REQUIRED_CHECKPOINT_FILES.issubset({item.name for item in path.iterdir()}):
        return False
    try:
        metadata = json.loads((path / "CARNAGE_METADATA.json").read_text(encoding="utf-8"))
        if metadata.get("metadata_schema_version") != 1 or not isinstance(metadata.get("compatibility"), dict):
            return False
        files = metadata.get("files")
        if not isinstance(files, dict):
            return False
        def matches_metadata(name: str) -> bool:
            metadata_name = name.upper() if name.endswith(".lt") else name
            record = files.get(metadata_name)
            return isinstance(record, dict) and record.get("size") == (path / name).stat().st_size
        return all(matches_metadata(name) for name in REQUIRED_CHECKPOINT_FILES - {"CARNAGE_METADATA.json"})
    except (OSError, ValueError, TypeError):
        return False


def restore(handle: str, checkpoint_root: Path) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(kagglehub.dataset_download(handle))
    except Exception as error:
        print(f"No checkpoint dataset restored ({error}); a fresh run may start.")
        return
    for source in downloaded.iterdir():
        if source.suffix == ".tar":
            with tarfile.open(source) as archive:
                destination = checkpoint_root.resolve()
                for member in archive.getmembers():
                    resolved = (checkpoint_root / member.name).resolve()
                    if destination not in resolved.parents and resolved != destination:
                        raise RuntimeError(f"Unsafe path in checkpoint archive: {member.name}")
                archive.extractall(checkpoint_root)
            continue
        target = checkpoint_root / source.name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    valid = [path.name for path in checkpoint_root.iterdir() if checkpoint_is_valid(path)]
    print(f"Restored checkpoints from {handle} to {checkpoint_root}; validated {len(valid)}: {valid}")


class CoalescedPublisher:
    def __init__(self, handle: str, checkpoint_root: Path, minimum_interval: int = 600) -> None:
        self.handle = handle
        self.checkpoint_root = checkpoint_root
        self.minimum_interval = minimum_interval
        self.last_marker = ""
        self.last_upload = 0.0
        self.pending = False
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None

    def marker(self) -> str:
        latest = self.checkpoint_root / "LATEST.json"
        return latest.read_text(encoding="utf-8") if latest.exists() else ""

    def request(self, force: bool = False) -> None:
        marker = self.marker()
        if not marker or (marker == self.last_marker and not force):
            return
        with self.lock:
            self.pending = True
            if self.worker and self.worker.is_alive():
                return
            self.worker = threading.Thread(target=self._run, args=(force,), daemon=False)
            self.worker.start()

    def _run(self, force: bool) -> None:
        while True:
            with self.lock:
                if not self.pending:
                    return
                self.pending = False
            wait = 0.0 if force else max(0.0, self.minimum_interval - (time.time() - self.last_upload))
            if wait:
                time.sleep(wait)
            marker = self.marker()
            if not marker:
                continue
            try:
                metadata = json.loads(marker)
                notes = f"Carnage checkpoint at {metadata.get('total_timesteps', 'unknown')} timesteps"
                self._upload(notes)
                self.last_marker = marker
                self.last_upload = time.time()
                print(f"Published {notes} to {self.handle}")
            except Exception as error:
                print(f"Checkpoint upload failed and will be retried: {error}", file=sys.stderr)
                with self.lock:
                    self.pending = True
                time.sleep(30)

    def _upload(self, notes: str) -> None:
        owner, slug = self.handle.split("/", 1)
        with tempfile.TemporaryDirectory(prefix="carnage-dataset-", dir=self.checkpoint_root.parent) as temporary:
            stage = Path(temporary)
            for checkpoint in self.checkpoint_root.iterdir():
                if not checkpoint.is_dir() or checkpoint.name.startswith("."):
                    continue
                if not checkpoint_is_valid(checkpoint):
                    print(f"Skipping invalid checkpoint during dataset publish: {checkpoint}", file=sys.stderr)
                    continue
                with tarfile.open(stage / f"{checkpoint.name}.tar", "w") as archive:
                    archive.add(checkpoint, arcname=checkpoint.name)
            latest = self.checkpoint_root / "LATEST.json"
            if latest.exists():
                shutil.copy2(latest, stage / latest.name)
            (stage / "dataset-metadata.json").write_text(json.dumps({
                "id": self.handle,
                "title": slug.replace("-", " ").title(),
                "isPrivate": True,
                "licenses": [{"name": "other"}],
            }), encoding="utf-8")

            exists = subprocess.run(
                ["kaggle", "datasets", "status", self.handle],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            command = [
                "kaggle", "datasets", "version" if exists else "create",
                "-p", str(stage), "-q", "-r", "skip",
            ]
            if exists:
                command += ["-m", notes, "-d"]
            subprocess.run(command, check=True)

    def finish(self) -> None:
        self.request(force=True)
        if self.worker:
            self.worker.join()


def run_with_sync(handle: str, checkpoint_root: Path, command: list[str]) -> int:
    publisher = CoalescedPublisher(handle, checkpoint_root)
    process = subprocess.Popen(command)

    def request_stop(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while process.poll() is None:
            publisher.request()
            time.sleep(30)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait()
        publisher.finish()
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    restore_parser = modes.add_parser("restore")
    restore_parser.add_argument("--handle", required=True)
    restore_parser.add_argument("--checkpoint-root", type=Path, required=True)
    run_parser = modes.add_parser("run")
    run_parser.add_argument("--handle", required=True)
    run_parser.add_argument("--checkpoint-root", type=Path, required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.mode == "restore":
        restore(arguments.handle, arguments.checkpoint_root)
        return 0
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("run mode requires a command after --")
    return run_with_sync(arguments.handle, arguments.checkpoint_root, command)


if __name__ == "__main__":
    raise SystemExit(main())
