from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="plaid",
        version=0.1,
        author="Amy X. Lu",
        author_email="lu.amy326@gmail.com",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
    )
