# MkopoGuard v2 — Devlog

## 2026-10-05 — Phase 0 setup

**Did:** Created the repo, virtual environment, folder structure, Ruff + pre-commit, a smoke test and the GitHub Actions CI pipeline.

**Broke / learned:** I learnt how to create tests and create a github actions CI pipeline . I made a mistake in the creation of the __init__py files which I was able to resolve by finding the position of the files . I also had an intial conflict in pushing the README file to github after configuring the github actions pipeline but I was able to solve it in good time

**Next:** Phase 1.1 — audit notebook for the 20-country Findex pool.


## 2026-10-06 — Phase 1 Auditing the dataset


**Did:** I was able to import the dataset to the project and audit it to find the required countries and columns I will be using for this project . I was able to perform tests on the dataset config file to ensure it has been appropriately set.

**Broke / learned:**
Learnt that about the package named as pyarrow which is used for interoperability between packages in python (numpy, pandas and others )
Learnt how to succesfully run notebooks on my machine
Used ISO country codes instead of names because "Côte d'Ivoire" gets mangled

**Decisions:**
- Choices I made and why (or "None")
