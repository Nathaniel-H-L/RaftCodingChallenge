from setuptools import setup, find_packages

setup(
    name="raft-langgraph-agent",
    version="0.1.0",
    description="LangGraph-based AI agent for parsing unstructured order data",
    author="Nathaniel Luscomb",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.0.40",
        "langchain>=0.1.16",
        "langchain-openai>=0.1.6",
        "requests>=2.31.0",
        "pydantic>=2.6.0",
        "python-dotenv>=1.0.1",
    ],
    entry_points={
        "console_scripts": [
            "raft-agent=raft_agent.main:main",
        ]
    },
)