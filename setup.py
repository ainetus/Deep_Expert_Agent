import os
import setuptools
from setuptools import setup

__version__ = '0.1.0'

# Set EXPERTAGENT_CPU_ONLY=1 to depend on the CPU-only TensorFlow wheel instead of the
# GPU-capable one. The Docker build sets it (see Dockerfile) because its target server
# has no GPU; leaving it unset keeps the default, GPU-capable install for everyone else.
#
# Only TensorFlow needs switching here: it ships CPU and GPU builds under *different*
# distribution names, so `pip install .` would otherwise pull the GPU `tensorflow` wheel
# straight back in on top of the `tensorflow-cpu` that requirements_docker.txt installed.
# torch uses the same name for both builds -- which one you get is decided by the index
# it is fetched from (see the --extra-index-url in requirements_docker.txt).
CPU_ONLY = os.environ.get("EXPERTAGENT_CPU_ONLY", "0") == "1"

pkgs = {
    "required": [
        "cvxpy",
        "grid2op",
        "l2rpn_baselines",
        "lightsim2grid",
        "numpy",
        "torch",
        "tensorflow-cpu" if CPU_ONLY else "tensorflow",
        "stable-baselines3",
        "imageio",
        "numba",
        "lxml",
    ],
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ExpertAgent',
      version=__version__,
      description='ExpertAgent to solve power grid congestion and overload',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.10',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='AI4REALNET project',
      author='IRTSX',
      url="https://github.com/AI4REALNET/T2.1_deep_expert",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={
            # If any package contains *.txt or *.rst files, include them:
            "": ["*.ini", "*.zip", "*.npz"],
            },
      install_requires=pkgs["required"],
    #   extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': []
     }
)