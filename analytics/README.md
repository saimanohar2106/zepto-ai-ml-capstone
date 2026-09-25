\# Module 2 — Analytics Pipeline



\## Overview



This module performs exploratory data analysis (EDA), preprocessing, classification, imbalance analysis, Random Forest tuning, and regression using the classic Titanic dataset.



The workflow is designed to be reproducible and follows a train-only preprocessing approach for machine learning.



\---



\# 1. Dataset Loading and Offline Fallback



The Titanic dataset was loaded once using:



```python

sns.load\_dataset("titanic")

