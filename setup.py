from setuptools import find_packages, setup

install_requires = []


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


if __name__ == "__main__":
    setup(
        name="qip",
        version="1.0.0",
        description="Quantum informed molecular pretraining model",
        long_description=readme(),
        long_description_content_type="text/markdown",
        author="Standigm",
        keywords="admet prediction, de novo drug discovery, drug discovery, drug design",
        url="https://github.com/Standigm/QIP",
        packages=find_packages(
            exclude=(
                "configs",
                "runs",
                "mlruns",
                "examples",
                "demo",
                "results",
                "outputs",
                "exps",
                "docs",
            )
        ),
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        # license='Apache License 2.0',
        python_requires=">=3.7.0",
        install_requires=install_requires,
        ext_modules=[],
        zip_safe=False,
    )
