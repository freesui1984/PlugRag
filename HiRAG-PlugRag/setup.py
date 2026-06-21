import setuptools
from setuptools import find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


vars2find = ["__author__", "__version__", "__url__"]
vars2readme = {}
with open("./hirag/__init__.py", encoding="utf-8") as f:
    for line in f.readlines():
        for v in vars2find:
            if line.startswith(v):
                line = line.replace(" ", "").replace('"', "").replace("'", "").strip()
                vars2readme[v] = line.split("=")[1]

deps = []
with open("./requirements.txt", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        deps.append(line)

setuptools.setup(
    name="HiRAG",
    url=vars2readme["__url__"],
    version=vars2readme["__version__"],
    author=vars2readme["__author__"],
    description="HiRAG implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=deps,
)
