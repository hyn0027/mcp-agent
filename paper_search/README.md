# 3 Collecting Agent Safety and Security Benchmarks: From “Guess What Safety Means” to Concrete Rules

`seed_papers.md` contains the seed paper list for system literature review.

`search.py` contains code where we use arXiv API to search for paper.

`filter.py` conains code where we select whether the paper propose benchmarks.

The json files are intermediate results, please refer to `filter.py` for details.

`FINAL_RES.csv` contains the final annotation result on the 301 papers and all detailed note for the 80 papers, as explained in section 3.

More readable versions of `FINAL_RES.csv` are published with google sheets at: <https://docs.google.com/spreadsheets/d/1JtqN-mgOP7MQJGIfLmJVdtwA1nJvFhaM5ZHWaTjAj_8/edit?usp=sharing> and Hugging Face at: <https://huggingface.co/datasets/hyn0027D/agent-symbolic-guardrails>.