import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import requests

# /NS/chatgpt/work/qwu/hallucinations_detection/code/eval/entity_knowledge/entity_hallucination_with_fastMCP/data/user_unique_data_sampled_500.csv
# Check the 'entity' column, search it on google using the SerpAPI
# Check: if the entity has wikipedia
# Check: how many search results we have for that entity
# read the SERP_API_PRIVATE through the environment
# Save the info in another two column and a new data csv file
# and then plot the distribution of the search results num (fig1) and print how many has wiki how many doesn't

DATA_PATH = "/NS/chatgpt/work/qwu/hallucinations_detection/code/eval/entity_knowledge/entity_hallucination_with_fastMCP/data/user_unique_data_sampled_500.csv"
OUTPUT_PATH = "/NS/chatgpt/work/qwu/hallucinations_detection/code/eval/entity_knowledge/analyse/entity_popularity_results.csv"
FIG_PATH = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/figures/search_results_distribution.pdf"

SERP_API_KEY = '.'



def search_entity(entity: str) -> dict:
    """Search entity on Google via SerpAPI, return has_wikipedia and total_results."""
    params = {
        "q": entity,
        "api_key": SERP_API_KEY,
        "engine": "google",
        "num": 10,
    }
    response = requests.get("https://serpapi.com/search", params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Total number of search results
    total_results = None
    search_info = data.get("search_information", {})
    raw = search_info.get("total_results")
    if raw is not None:
        try:
            total_results = int(str(raw).replace(",", "").replace(".", "").split()[0])
        except (ValueError, IndexError):
            total_results = None

    # Check if any organic result links to Wikipedia
    has_wikipedia = False
    for result in data.get("organic_results", []):
        link = result.get("link", "")
        if "wikipedia.org" in link:
            has_wikipedia = True
            break

    return {"has_wikipedia": has_wikipedia, "total_results": total_results}


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Found existing results, loading from {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
        # Support both 'entity' and 'entity_text' column names
        entity_col = "entity" if "entity" in df.columns else "entity_text"

        has_wikipedia_list = []
        total_results_list = []

        for i, entity in enumerate(df[entity_col]):
            print(f"[{i+1}/{len(df)}] Searching: {entity}")
            try:
                result = search_entity(str(entity))
                has_wikipedia_list.append(result["has_wikipedia"])
                total_results_list.append(result["total_results"])
            except Exception as e:
                print(f"  ERROR: {e}")
                has_wikipedia_list.append(None)
                total_results_list.append(None)
            # Be polite to the API
            time.sleep(0.5)

        df["has_wikipedia"] = has_wikipedia_list
        df["total_results"] = total_results_list

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved results to {OUTPUT_PATH}")

    # --- Stats: Wikipedia ---
    wiki_counts = df["has_wikipedia"].value_counts(dropna=False)
    has_wiki = int(wiki_counts.get(True, 0))
    no_wiki = int(wiki_counts.get(False, 0))
    print(f"\nWikipedia presence:")
    print(f"  Has Wikipedia : {has_wiki}")
    print(f"  No Wikipedia  : {no_wiki}")

    # --- Fig 1: Violin plot of total_results (all entities) ---
    # Remove outliers using IQR (1.5x rule) before plotting
    arr = df["total_results"].dropna()
    q1, q3 = arr.quantile(0.25), arr.quantile(0.75)
    iqr = q3 - q1
    data_to_plot = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)].tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    parts = ax.violinplot([data_to_plot], positions=[0], showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    ax.set_xticks([])
    ax.set_ylabel("Num of Google Search Results", fontsize=25)
    # ax.set_title("Distribution of Search Result Counts", fontsize=25)
    ax.tick_params(axis="y", labelsize=25)
    plt.tight_layout()
    plt.savefig(FIG_PATH)
    print(f"Saved figure to {FIG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
