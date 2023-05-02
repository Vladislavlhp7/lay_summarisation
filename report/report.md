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
  - \usepackage[inline]{enumitem}
  - \bibliographystyle{acl_natbib.bst}
abstract: |
  In this work we design an extractive-abstractive lay summarisation pipeline for biomedical papers [@biolaysumm-2023-overview] that generates summaries for non-experts.
  For that purpose, we create a sentence-level ROUGE-maximising dataset from the gold summaries and the whole articles, which we then use to train a BERT-based classifier to identify the most important sentences per article.
  Once an extracted summary is produced we feed it into two abstractive models (Clinical-Longformer [@li2023comparative], GPT-2 [@radford2019language]) that paraphrase the summary into a more readable version.
  To evaluate our models we used the ROUGE metric [@lin-2004-rouge] and the readability metrics - FKGL [@Kincaid1975DerivationON], Gunning Fog Score [@gunning1952technique], and ARI [@senter1967automated] on the gold summaries and the generated summaries.
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
\begin{table}[htbp]
    \centering
    \begin{tabular}{|c|c|c|}
        \hline
        \textbf{Dataset} & \textbf{Avg. Sentences} & \textbf{Avg. Tokens} \\
        \hline
            PLOS & $300$ & $9,000$ \\
        \hline
            eLife & $600$ & $14,000$ \\
        \hline
    \end{tabular}
    \caption{PLOS and eLife: Dataset statistics}\label{tab:dataset_stats}
\end{table}
## Extractor Network {#sec:extractor-network}
Due to the extreme length of medical articles (e.g., eLife has an average of 600 sentences per article), 
it is not feasible to pass them directly as input to the abstractive models due to their limited maximum input size:

i. GPT-2 [@radford2019language]: $1,024$ tokens, and
ii. Clinical-Longformer [@li2023comparative]: $4,096$ tokens

To overcome this limitation, we use the BioClinicalBERT [@alsentzer-etal-2019-publicly] model, pre-trained on the MIMIC-III dataset [@Johnson2016MIMICIII],
to extract the most important sentences from the articles.
For that purpose, we cast the extraction summarisation problem as supervised binary classification where the input is a sentence $s$ 
and the output is a binary label indicating whether the sentence should be included in the summary $c$ or not (i.e., 1 and 0, respectively).
Due to the nature of the provided gold summaries (i.e., abstractive and lay), we generate our own sentence-level 
dataset by applying the ROUGE-maximisation technique [@zmandar-etal-2021-joint, @nallapati2017summarunner] on the gold summaries and the whole articles. 
More formally, for each gold summary sentence $s_{i}^{k}$, we find the sentence $s_{j}^{k}$ in article $a_{k}$ that maximises the ROUGE-2 score between them.
We then label $s_{j}^{k}$ as 1 and the rest of the sentences in $a_{k}$ as 0.
Because the number of sentences in the articles is much larger than the number of sentences in the gold summaries:

i. We base our extractive binary dataset on both eLife and PLOS data to maximise the number of training samples;
ii. We further resolve the class imbalance problem by random under-sampling the majority class (i.e., $0$) 
          to match the number of samples in the minority class (i.e., $1$);

Our final extractive training dataset consists of $944,234$ sentences with a completely balanced class distribution.


# Evaluation {#sec:evaluation}

# Discussion and Conclusion {#sec:discussion-conclusion}

# Bibliography
