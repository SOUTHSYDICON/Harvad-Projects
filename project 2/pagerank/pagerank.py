import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pages = list(corpus)
    total = len(pages)
    links = corpus[page]
    distribution = {}
    if not links:
        for p in pages:
            distribution[p] = 1 / total
        return distribution
    for p in pages:
        distribution[p] = (1 - damping_factor) / total
        if p in links:
            distribution[p] += damping_factor / len(links)
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus)
    counts = {p: 0 for p in pages}
    current = random.choice(pages)
    counts[current] += 1
    for _ in range(n - 1):
        model = transition_model(corpus, current, damping_factor)
        current = random.choices(list(model), weights=list(model.values()))[0]
        counts[current] += 1
    return {p: counts[p] / n for p in pages}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus)
    total = len(pages)
    ranks = {p: 1 / total for p in pages}
    while True:
        new_ranks = {}
        for p in pages:
            incoming = 0
            for other in pages:
                if not corpus[other]:
                    incoming += ranks[other] / total
                elif p in corpus[other]:
                    incoming += ranks[other] / len(corpus[other])
            new_ranks[p] = (1 - damping_factor) / total + damping_factor * incoming
        converged = all(abs(new_ranks[p] - ranks[p]) <= 0.001 for p in pages)
        ranks = new_ranks
        if converged:
            break
    return ranks


if __name__ == "__main__":
    main()
