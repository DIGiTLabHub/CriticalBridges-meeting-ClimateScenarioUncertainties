from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"
REQUIREMENTS = ROOT / "requirements.txt"


def read_readme() -> str:
    return README.read_text(encoding="utf-8") if README.exists() else ""


def read_requirements() -> list[str]:
    if not REQUIREMENTS.exists():
        return []
    return [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="critical-bridges-climate-uncertainties",
    version="0.3.0",
    description="Climate-scenario-aware bridge scour simulation and surrogate modeling workflow.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=find_packages(
        include=[
            "src",
            "src.*",
            "config",
            "config.*",
            "BridgeModeling",
            "BridgeModeling.*",
        ]
    ),
    install_requires=read_requirements(),
    include_package_data=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
)
