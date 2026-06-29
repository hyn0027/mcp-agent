import csv
import json
import os
import asyncio
import random
from openai import AsyncOpenAI


async def llm_annotate(
    sys_prompt, user_prompt, papers, label_key, prompt_cache_key, model="gpt-5-nano"
) -> list:
    client = AsyncOpenAI()

    MAX_CONCURRENCY = 30

    async def process_paper(paper) -> str:
        title = paper["title"]
        abstract = paper["summary"]
        prompt = sys_prompt + "\n" + user_prompt.format(title=title, abstract=abstract)

        response = await client.responses.create(
            model=model,
            input=prompt,
            prompt_cache_key=prompt_cache_key,
        )
        return response.output_text.strip()

    async def process_one(paper, sem):
        async with sem:
            try:
                result = await process_paper(paper)
            except Exception:
                result = "-1"
            paper[label_key] = result
            return paper

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [process_one(paper, sem) for paper in papers]
    return await asyncio.gather(*tasks)


async def llm_annotate_paper_is_benchmark(papers) -> list:
    system_prompt = (
        "You are an assistant that helps user filter papers based on their title and abstract. "
        "For the paper title and abstract, you will determine whether the paper propose a benchmark or dataset. "
        "If the paper propose a benchmark or dataset, you will return 1, otherwise, you will return 0. "
        "If you are unsure about the paper, you will return -1.\n"
        "You should output a single number (1, 0, or -1) without any explanation or additional text."
    )
    user_prompt = "Title: {title}\nAbstract: {abstract}\n"

    return await llm_annotate(
        sys_prompt=system_prompt,
        user_prompt=user_prompt,
        papers=papers,
        label_key="llm_annotate_is_benchmark",
        prompt_cache_key="prompt_cache_key_filter_paper",
    )


def print_paper_info(paper, keywords) -> None:
    title = paper["title"]
    abstract = paper["summary"]
    pdf_url = paper.get("pdf_url", "N/A")
    paper_info = f"\033[91mTitle:\033[0m {title}\n\033[91mAbstract:\033[0m {abstract}\n\033[91mPDF URL:\033[0m {pdf_url}\n"
    for keyword in keywords:
        # change to red color regardless of case
        paper_info = paper_info.replace(keyword, f"\033[91m{keyword}\033[0m")
        paper_info = paper_info.replace(
            keyword.capitalize(), f"\033[91m{keyword.capitalize()}\033[0m"
        )
        paper_info = paper_info.replace(
            keyword.upper(), f"\033[91m{keyword.upper()}\033[0m"
        )

    print(paper_info)


def human_label(
    papers, num_label, llm_key, human_key, request_human_label_info, keywords
) -> list:
    def request_human_label(paper) -> str:
        if human_key in paper:
            return paper[human_key]
        print_paper_info(paper, keywords)
        label = input(request_human_label_info)
        while label not in ["1", "0", "-1"]:
            label = input(
                "Invalid input. Please enter 1 for yes, 0 for no, or -1 for unsure: "
            )
        return label

    def compute_agreement(papers) -> None:
        pairs = [
            (int(paper[llm_key]), int(paper[human_key]))
            for paper in papers
            if human_key in paper and int(paper[llm_key]) != -1
        ]
        if not pairs:
            print("No papers have human labels.")
            return
        print("-" * 50)
        print(f"Total papers with human labels: {len(pairs)}")

        # 95% confidence interval for kappa
        import numpy as np
        from statsmodels.stats.inter_rater import cohens_kappa, to_table

        data = np.asarray(pairs, dtype=int)  # shape should be (n, 2)

        table, bin = to_table(data)
        res = cohens_kappa(table)

        print(f"kappa = {res.kappa:.2f}")
        print(f"SE = {res.std_kappa:.2f}")
        print(f"95% CI = [{res.kappa_low:.2f}, {res.kappa_upp:.2f}]")

        confusion_matrix = {
            "TP": sum(1 for pair in pairs if pair[0] == 1 and pair[1] == 1),
            "TN": sum(1 for pair in pairs if pair[0] == 0 and pair[1] == 0),
            "FP": sum(1 for pair in pairs if pair[0] == 1 and pair[1] == 0),
            "FN": sum(1 for pair in pairs if pair[0] == 0 and pair[1] == 1),
        }
        precision = (
            confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FP"])
            if (confusion_matrix["TP"] + confusion_matrix["FP"]) > 0
            else 0
        )
        recall = (
            confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"])
            if (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0
            else 0
        )
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"Confusion Matrix: {confusion_matrix}")
        print(
            f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}"
        )

        ground_truth_positive = sum(1 for pair in pairs if pair[1] == 1)
        ground_truth_negative = sum(1 for pair in pairs if pair[1] == 0)
        print(f"Ground Truth Positive Ratio: {ground_truth_positive / len(pairs):.2f}")
        print(f"Ground Truth Negative Ratio: {ground_truth_negative / len(pairs):.2f}")
        labeled_positive = sum(int(paper[llm_key]) for paper in papers)
        labeled_negative = sum(1 - int(paper[llm_key]) for paper in papers)
        print(f"Labeled Positive Ratio: {labeled_positive / len(papers):.2f}")
        print(f"Labeled Negative Ratio: {labeled_negative / len(papers):.2f}")
        print("-" * 50)

    # label -1
    for paper in papers:
        if llm_key not in paper:
            paper[llm_key] = "-1"
        if paper[llm_key] == "-1" and human_key not in paper:
            print_paper_info(paper, keywords)
            label = input(request_human_label_info)
            while label not in ["1", "0", "-1"]:
                label = input(
                    "Invalid input. Please enter 1 for yes, 0 for no, or -1 for unsure: "
                )
            paper[human_key] = label

    unlabeled_paper_idxs = [
        idx for idx, paper in enumerate(papers) if human_key not in paper
    ]
    random_ids = random.sample(
        unlabeled_paper_idxs, min(num_label, len(unlabeled_paper_idxs))
    )
    for idx in random_ids:
        paper = papers[idx]
        label = request_human_label(paper)
        paper[human_key] = label
        with open("human_labeled_papers.json", "w") as f:
            json.dump(papers, f, indent=4)
    compute_agreement(papers)
    return papers


def human_label_paper_is_benchmark(papers, num_label) -> list:

    return human_label(
        papers,
        num_label,
        llm_key="llm_annotate_is_benchmark",
        human_key="human_annotate_is_benchmark",
        request_human_label_info="Does this paper propose a benchmark or dataset? Enter 1 for yes, 0 for no, or -1 for unsure: ",
        keywords=[
            "agent",
            "bench",
            "dataset",
            "framework",
            "propose",
            "eval",
            "assess",
        ],
    )


def main() -> None:
    path = "filtered_papers.json"
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Total papers to label: {len(data)}")

    if os.path.exists("filtered_paper_with_results.json"):
        with open("filtered_paper_with_results.json", "r") as f:
            results = json.load(f)
    else:
        results = asyncio.run(llm_annotate_paper_is_benchmark(data))
        with open("filtered_paper_with_results.json", "w") as f:
            json.dump(results, f, indent=4)

    label_counts = {}
    for paper in results:
        label = paper.get("llm_annotate_is_benchmark", "-1")
        label_counts[label] = label_counts.get(label, 0) + 1
    print("Label distribution for benchmark/dataset:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} papers")

    if os.path.exists("human_labeled_papers.json"):
        with open("human_labeled_papers.json", "r") as f:
            results = json.load(f)
        results = human_label_paper_is_benchmark(results, num_label=0)
    else:
        results = human_label_paper_is_benchmark(results, num_label=100)
        with open("human_labeled_papers.json", "w") as f:
            json.dump(results, f, indent=4)

    labeled_benchmark_papers = []
    for paper in results:
        if (
            "human_annotate_is_benchmark" in paper
            and paper["human_annotate_is_benchmark"] == "1"
        ):
            labeled_benchmark_papers.append(paper)
        elif (
            "human_annotate_is_benchmark" not in paper
            and paper["llm_annotate_is_benchmark"] == "1"
        ):
            labeled_benchmark_papers.append(paper)
    with open("labeled_benchmark_papers.json", "w") as f:
        json.dump(labeled_benchmark_papers, f, indent=4)
    results = labeled_benchmark_papers
    print(f"Total papers labeled as benchmark/dataset: {len(results)}")

    # output csv, with title and link
    with open("final_paper_list.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Title", "Link", "Date"])
        for idx, paper in enumerate(results):
            writer.writerow(
                [
                    idx,
                    paper.get("title", "N/A"),
                    paper.get("pdf_url", "N/A"),
                    paper.get("published", "N/A"),
                ]
            )


if __name__ == "__main__":
    main()
