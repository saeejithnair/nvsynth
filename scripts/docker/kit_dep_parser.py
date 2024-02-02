import tomli
from pathlib import Path

from typing import Iterable, Callable


EXT_DIRS: list[tuple[Path, bool]] = [
    (Path("/isaac-sim/exts"), True),
    (Path("/isaac-sim/kit/exts"), True),
    (Path("/isaac-sim/kit/extscore"), True),
    (Path("/isaac-sim/extscache"), False),
    (Path("/isaac-sim/extsPhysics"), False)
]


def filter_for_key(filepath: str, top_level_key: str) -> str:
    """Parses toml-like .kit files, returning all elements in a specific top-level key.

    Args:
        filepath (str): path to the .kit file.
        top_level_key (str): top level key whose elements are to be found.

    Returns:
        str: valid toml file as string with single top-level key
    """
    top_level_key = f"[{top_level_key}]"
    ret = [top_level_key]

    with open(filepath) as f:
        is_top_level_key = False
        for line in f.readlines():
            if line.startswith("["):
                is_top_level_key = line.startswith(top_level_key)
            elif is_top_level_key:
                ret.append(line)

    return "\n".join(ret)


def find_package_paths(pkg_names: list[str]) -> list[str]:
    """Find the paths to the packages, looking in predetermined extension directories.

    Args:
        pkg_names (list[str]): names of packages

    Returns:
        list[str]: paths to packages.
    """
    dir_names = []

    for dir_name in pkg_names:
        for parent_dir, no_ver in EXT_DIRS:
            if no_ver:
                dir = parent_dir / dir_name
                if dir.exists() and dir.is_dir():
                    dir_names.append(str(dir))
            else:
                for dir in parent_dir.glob(f"{dir_name}*"):
                    if dir.exists() and dir.is_dir():
                        dir_names.append(str(dir))

    return dir_names


PROCESS: Callable[[Iterable[str]], str] = {
    "pythonpath": lambda vals: ":".join(vals),
    "pyright": lambda vals: "{\n  \"extraPaths\": [\n" + ",\n    ".join([f'"{i}"' for i in vals]) + "\n  ]\n}"
}


def main(filepath='/nvsynth/.configs/isaac-sim/omni.isaac.sim.python.kit', top_level_key="dependencies", process_type="pyright"):
    tomls = filter_for_key(filepath, top_level_key)
    pkg_names = tomli.loads(tomls)[top_level_key].keys()
    pkg_dirs = find_package_paths(pkg_names)
    process = PROCESS[process_type]
    print(process(pkg_dirs))


if __name__ == "__main__":
    main()
