import os
import setuptools

current_directory = os.path.dirname(os.path.realpath(__file__))
requirement_path = os.path.join(current_directory,
                               "requirements.txt")

install_requires = []
with open(requirement_path) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="src_trust_learning",
    version="0.1",
    author="Resul Dagdanov",
    author_email="",
    description="source codes for learning trust behavior via IRL",
    url="https://github.com/resuldagdanov/learning-trust-behavior",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
)
