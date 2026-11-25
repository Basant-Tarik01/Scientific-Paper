from setuptools import find_packages, setup

package_name = "amr_sim_nodes"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="warehouse-amr-sim team",
    maintainer_email="maintainer@example.com",
    description="ROS 2 nodes: fault injection, pedestrian simulation, metrics, data logging, Nav2 episode runner.",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "fault_injector = amr_sim_nodes.fault_injector:main",
        "pedestrian_sim = amr_sim_nodes.pedestrian_sim:main",
        "metrics_monitor = amr_sim_nodes.metrics_monitor:main",
        "data_logger = amr_sim_nodes.data_logger:main",
        "episode_runner = amr_sim_nodes.episode_runner:main",
        "gt_localization = amr_sim_nodes.gt_localization:main",
    ]},
)
