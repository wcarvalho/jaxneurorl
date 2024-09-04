
from setuptools import setup, find_packages

setup(
    name="JaxNeuroRL",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "wandb",
        "xminigrid==0.6.0",
        "ipdb",
        "tqdm==4.66.2",
        "rlax==0.1.6",
        "mctx==0.0.5",
        "ray[tune]==2.10.0",
        "typing_extensions==4.11.0",
        "optax==0.2.1",
        "scipy==1.12.0",
    ],
    author="Wilka Carvalho",
    author_email="wcarvalho92@gmail.com",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wcarvalho/jaxneurorl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
