# [Title of Your Master's Thesis]

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow" alt="Status">
  <img src="https://img.shields.io/github/license/username/repo" alt="License">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
</p>

---

## 📌 Project Presentation
Provide a high-level summary of your research. What is the core problem? What is your methodology? 

> **Example:** This repository contains the experimental framework for my Master's Thesis. The project focuses on analyzing Binance market volatility using [Specific Model/Algorithm]. It bridges the gap between high-frequency trading data and [Theoretical Concept].

* **Objective:** To evaluate $X$ in the context of $Y$.
* **Key Contribution:** Implementation of a custom pipeline for $Z$.
* **Live Site:** [View the GitHub Pages Dashboard](https://yourusername.github.io/your-repo-name/)

---

## 📝 Related Papers
List the foundational literature your thesis builds upon. If your own paper is published/pre-printed, put it first.

1.  **Author, A., & Author, B. (202X).** *Title of the most relevant paper*. Journal Name. [Link]
2.  **Binance Research.** *Market Microstructure Report*. [Link]
3.  **Your Name (2026).** *Preliminary Findings on Crypto Volatility*. Thesis Working Paper.

---

## 📂 Repo Structure
A clear map helps others (and your supervisor) navigate your work.

```text
├── data/               # Raw and processed datasets (ignored if large)
├── docs/               # Documentation and GitHub Pages assets
├── models/             # Trained model checkpoints or logic
├── notebooks/          # Jupyter notebooks for exploratory analysis
├── scripts/            # Utility Python scripts
├── .env.example        # Template for Binance API keys
├── .gitignore          # Prevents .env and data leaks
├── main.py             # Entry point of the application
└── requirements.txt    # Project dependencies

## Installation

pip install -r requirements.txt

jax 0.6.2 doesn't behave very well with nvidia-cublas-cu12 12.8, but torch requires this version. Hence, we need to overwrite the package with the newer version (no real issue).
pip install --no-deps --force-reinstall nvidia-cublas-cu12>=12.9.1.4