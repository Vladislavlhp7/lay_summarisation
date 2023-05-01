---
title: Lay It On Me. Generating Easy-to-Read Summaries for Non-Experts
author:
  - Ahmed Soliman
  - Marc Wenzlawski
  - Vladislav Yotkov
bibliography: custom.bib
header-includes:
  - \usepackage{acl}
  - \usepackage{natbib}
  - \bibliographystyle{acl_natbib.bst}
abstract: |
  This is the abstract.
  In this work we design an extractive-abstractive lay summarisation pipeline for biomedical papers[@biolaysumm-2023-overview] that generates summaries for non-experts.
  For that purpose, we create a sentence-level ROUGE-maximising dataset from the gold summaries and the whole articles, which we then use to train a BERT-based classifier to identify the most important sentences per article.
  Once an extracted summary is produced we feed it into three abstractive models (Clinical-T5[@lehman2023clinical], Clinical-Longformer[@li2023comparative], GPT-2[@radford2019language]) that paraphrase the summary into a more readable version.
  To evaluate our models we used the ROUGE metric[@lin-2004-rouge] and the readability metrics - FKGL[@Kincaid1975DerivationON], Gunning Fog Score[@gunning1952technique], and ARI[@senter1967automated] on the gold summaries and the generated summaries.
  Results ...
---

# Introduction {#sec:introduction}

# Related Work {#sec:related-work}

# Methods and Datasets {#sec:methods}
## Dataset {#sec:dataset}
\begin{table}[htbp]
    \centering
    \begin{tabular}{|c|c|c|}
        \hline
        \textbf{Dataset} & \textbf{Training} & \textbf{Validation} \\
        \hline
        PLOS & $24,773$ & $1,376$ \\
        \hline
        eLife & $4,346$ & $241$ \\
        \hline
    \end{tabular}
    \caption{PLOS and eLife: number of articles}\label{tab:dataset_stats}
\end{table}
## Extractor Network {#sec:extractor-network}
Due to the extreme length of medical articles, it is not feasible to pass them directly as input to the abstractive models due to their limited maximum input size:
\begin{enumerate}[label=(\roman*)]
    \item Clinical-T5~\cite{lehman2023clinical}: $512$ tokens
    \item GPT-2~\cite{radford2019language}: $1,024$ tokens
    \item Clinical-Longformer~\cite{li2023comparative}: $4,096$ tokens
\end{enumerate}
# Evaluation {#sec:evaluation}

# Discussion and Conclusion {#sec:discussion-conclusion}

# Bibliography
