"""amr_sim_core is both a ROS 2 (ament_python) package and a plain pip package:
    colcon build --packages-select amr_sim_core      # inside the workspace
    pip install -e src/amr_sim_core                  # without ROS (2D simulator / gym env only)
"""
import os
from glob import glob

from setuptools import find_packages, setup

package_name = "amr_sim_core"
data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]
for sub in ("robot", "worlds", "scenarios"):
    data_files.append((os.path.join("share", package_name, "data", sub), glob(f"{package_name}/data/{sub}/*.yaml")))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["data/*/*.yaml"]},
    include_package_data=True,
    data_files=data_files,
    install_requires=["setuptools", "numpy", "pyyaml"],
    extras_require={"gym": ["gymnasium>=0.29"], "viz": ["matplotlib"],
                    "rl": ["gymnasium>=0.29", "stable-baselines3>=2.0", "sb3-contrib>=2.0", "torch"]},
    zip_safe=True,
    maintainer="warehouse-amr-sim team",
    maintainer_email="maintainer@example.com",
    description="World, robot, fault, pedestrian and metric models + fast 2D simulator and Gymnasium env "
                "for warehouse AMR navigation among humans under robot degradation.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "generate_assets = amr_sim_core.tools.generate_assets:main",
    ]},
)
