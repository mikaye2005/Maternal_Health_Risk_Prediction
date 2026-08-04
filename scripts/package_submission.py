from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "MamaCare_Final_Submission.zip"
ARCHIVE_ROOT = PurePosixPath("MamaCare_Final_Submission")
REPOSITORY_URL = "https://github.com/mikaye2005/Maternal_Health_Risk_Prediction"
BLOCKED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "dist"}
BLOCKED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        if BLOCKED_PARTS.intersection(relative.parts) or relative.suffix.lower() in BLOCKED_SUFFIXES:
            raise RuntimeError(f"Blocked tracked path cannot be packaged: {relative}")
        if relative.name == "CODEX_PROMPT.md":
            raise RuntimeError("Internal execution prompts must not be packaged.")
        full_path = ROOT / relative
        if not full_path.is_file():
            raise FileNotFoundError(f"Tracked file is missing: {relative}")
        paths.append(relative)
    return sorted(paths, key=lambda item: item.as_posix().lower())


def build_archive(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = tracked_files()
    readme_first = (
        "MamaCare Final Capstone Submission\n"
        "==================================\n\n"
        "Start with source_project/README.md. The source project contains the runnable code, "
        "executed notebook, model evidence, report, Responsible AI statement, presentation, "
        "tests and Streamlit demonstration.\n\n"
        "This is an academic classification prototype, not a diagnosis, treatment, triage or "
        "emergency-decision system.\n"
    ).encode("utf-8")
    manifest_name = (ARCHIVE_ROOT / "SUBMISSION_MANIFEST.txt").as_posix()
    checksum_name = (ARCHIVE_ROOT / "SHA256SUMS.txt").as_posix()
    payload: dict[str, bytes] = {
        (ARCHIVE_ROOT / "README_FIRST.txt").as_posix(): readme_first,
        (ARCHIVE_ROOT / "GITHUB_URL.txt").as_posix(): f"{REPOSITORY_URL}\n".encode("utf-8"),
        (ARCHIVE_ROOT / "SUBMISSION_CHECKLIST.md").as_posix(): (
            ROOT / "docs" / "SUBMISSION_CHECKLIST.md"
        ).read_bytes(),
    }

    for relative in files:
        data = (ROOT / relative).read_bytes()
        source_name = (ARCHIVE_ROOT / "source_project" / PurePosixPath(relative.as_posix())).as_posix()
        payload[source_name] = data

        convenience_name = None
        if relative.parts and relative.parts[0] == "reports":
            convenience_name = ARCHIVE_ROOT / PurePosixPath(relative.as_posix())
        elif relative.parts and relative.parts[0] == "presentation":
            convenience_name = ARCHIVE_ROOT / PurePosixPath(relative.as_posix())
        elif relative.parts[:2] == ("assets", "screenshots"):
            convenience_name = ARCHIVE_ROOT / "screenshots" / PurePosixPath(*relative.parts[2:])
        if convenience_name is not None:
            payload[convenience_name.as_posix()] = data

    all_archive_names = sorted([*payload, manifest_name, checksum_name])
    manifest_lines = [
        "MamaCare final submission payload",
        f"Public repository: {REPOSITORY_URL}",
        f"Tracked source-project files: {len(files)}",
        f"Archive files: {len(all_archive_names)}",
        "",
        "Archive paths:",
        *all_archive_names,
        "",
    ]
    payload[manifest_name] = "\n".join(manifest_lines).encode("utf-8")
    checksums = {name: sha256_bytes(data) for name, data in payload.items()}
    checksum_text = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(checksums.items())
    ).encode("utf-8")

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_name, data in sorted(payload.items()):
            archive.writestr(archive_name, data)
        archive.writestr(checksum_name, checksum_text)

    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha256_bytes(output.read_bytes())}  {output.name}\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the clean MamaCare final-submission archive.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    archive = build_archive(args.output)
    print(archive)


if __name__ == "__main__":
    main()
