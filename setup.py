from setuptools import setup, find_packages

setup(
    name="empathy_toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "nltk>=3.8",
        "flask>=2.3.0",
        "httpx>=0.24.0",  # Used for direct API calls instead of OpenAI SDK
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    author="Morgan Roberts MSW LPHA",
    author_email="morgan@forestwithintherapy.com",
    description="A toolkit for empathy modeling, dataset processing, and language model fine-tuning",
    keywords="empathy, nlp, machine learning, dataset",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
