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
  - \usepackage[justification=centering]{caption}
  - \bibliographystyle{acl_natbib.bst}
  - \setcounter{secnumdepth}{5}
graphics: yes
geometry:
    - top=15mm
    - bottom=15mm
    - left=15mm
    - right=15mm
abstract: |
    In this project, we present an extractive-abstractive lay summarization pipeline for biomedical papers aimed at generating accessible summaries for non-experts. To achieve this, we construct a sentence-level dataset optimized for maximizing ROUGE scores, utilizing both lay summaries and full articles. We employ a BERT-based classifier for identifying the most important sentences within each article. The extracted summaries are then input into two abstractive models, Clinical-Longformer and GPT-2, which paraphrase the summaries to enhance readability. We evaluate the performance of our models using the ROUGE metric, along with readability metrics such as Flesch-Kincaid Grade Level (FKGL), Gunning Fog Score, and Automated Readability Index (ARI). 
    We find that a ROUGE-maximizing extractive summarization approach is effective for generating extractive summaries, with the Clinical-Longformer model achieving the best results for combined ROUGE and readability scores.
    Our approach demonstrates the potential for generating lay-friendly summaries of biomedical papers, bridging the gap between expert knowledge and public understanding.
---

# Introduction {#sec:introduction}
Comprehending biomedical scientific publications can be difficult for non-experts, potentially leading to misinformed health decisions [@islam]. Lay summaries, simplified explanations of complex scientific content, could be a solution, but they are not always available. Despite past challenges in applying Automatic Text Summarisation (ATS) to biomedicine due to insufficient data [@chandrasekaran], two new datasets, PLOS and eLife, offer an opportunity to bridge this gap [@goldsack]. This study investigates ATS techniques for generating biomedical lay summaries using these datasets.

# Methods and Datasets {#sec:methods}

In this section outline the various ATS methodologies employed in this study and describe the PLOS and eLife datasets used for training and evaluation purposes.

## Dataset {#sec:dataset}

The data we used is sourced from biomedical research articles in English published in the Public Library of Science (PLOS) and eLife [@goldsack]. 
The datasets (Tables \ref{tab:dataset_stats_1} and \ref{tab:dataset_stats_2}) contain technical abstracts and lay summaries written by experts, which are part of BioLaySumm2023 shared task [@biolaysumm-2023-overview].

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
    \caption{PLOS and eLife: number of articles}\label{tab:dataset_stats_1}
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
    \caption{PLOS and eLife: Dataset statistics}\label{tab:dataset_stats_2}
\end{table}

## Extractor Network {#sec:extractor-network}

Due to the extreme length of medical articles (e.g., eLife has an average of $600$ sentences per article), 
it is not feasible to pass them directly as input to the abstractive models due to their limited maximum input size:

i. **GPT-2** [@radford2019language]: $1,024$ tokens, and
ii. **Clinical-Longformer** [@li2023comparative]: $4,096$ tokens

To overcome this limitation, we use the BioClinicalBERT [@alsentzer-etal-2019-publicly] model, pre-trained on the MIMIC-III dataset [@Johnson2016MIMICIII],
to extract the most important sentences from the articles.
For that purpose, we cast the extraction summarisation problem as supervised binary classification where the input is a sentence $s$ 
and the output is a binary label indicating whether the sentence should be included in the summary $c$ or not (i.e., $1$ and $0$, respectively).
Due to the nature of the provided gold summaries (i.e., abstractive and lay), we generate our own sentence-level 
dataset by applying the ROUGE-maximisation technique [@zmandar-etal-2021-joint;@nallapati2017summarunner] on the gold summaries and the whole articles. 
More formally, for each gold summary sentence $s_{i}^{k}$, we find the sentence $s_{j}^{k}$ in article $a_{k}$ that maximises the ROUGE-2 score between them.
We then label $s_{j}^{k}$ as $1$ and the rest of the sentences in $a_{k}$ as $0$.
Because the number of sentences in the articles is much larger than the number of sentences in the gold summaries:

i. We base our extractive binary dataset on both eLife and PLOS data to maximise the number of training samples;
ii. We further resolve the class imbalance problem by random under-sampling the majority class (i.e., $0$) 
          to match the number of samples in the minority class (i.e., $1$);

Our final extractive dataset consists of $944,234$ sentences with a completely balanced class distribution.
Data is further split into $80\%$-training, $10\%$-validation and $10\%$-testing datasets in a random stratified manner.
We then fine-tune the extractive model with a batch size of $32$ and a learning rate of $2 \times 10^{-5}$ following the guidance from 
BERT's authors [@Devlin2019BERTPO] and find that the model starts to over-fit beyond $2$ epochs 
(see Figures \ref{fig:extractor-eval-f1} and \ref{fig:extractor-eval-loss}).
We also report high F1 scores of $0.767$ and $0.765$ on the validation and test sets, respectively.

\begin{figure}
    \centering
    \includegraphics[width=0.5\textwidth]{charts/extractor-eval-f1.png}
    \caption{BioClinicalBERT: Evaluation F1}\label{fig:extractor-eval-f1}
    \vspace{0.25cm}
    \includegraphics[width=0.5\textwidth]{charts/extractor-eval-loss.png}
    \caption{BioClinicalBERT: Evaluation Loss}\label{fig:extractor-eval-loss}
\end{figure}

We then use the BioClinicalBERT model to predict the probability of each sentence in the article being _summarising_.
The top $10$ with the highest probability are selected and concatenated to produce the final extractive summary.
We arrive at this number after analysing the token distribution and finding that $10$ sentences is a reasonable number 
to fit within the maximum input size of the GPT-2 abstractive model (i.e., $1,024$ tokens split between the ten sentences and their lay paraphrases).
We also experiment with a top-$15$ strategy only for the Clinical Longformer to fully make use of the sparse attention mechanism (see Section \ref{sec:evaluation-quantitative}).
While we are aware that this can cause the _dangling anaphora phenomenon_ [@lin2009summarization], we use the 
extracted text only as an intermediate step fed into the abstractive models which paraphrase it into lay language.

## Abstractive Network {#sec:abstractive-network}

Once the extractive summary is generated, we train the abstractive models on the lay summaries and the extractive summaries. For this, we compare two models: GPT-2 [@radford2019language] and Clinical-Longformer [@li2022clinicallongformer]. 
We fine tune both models separately on eLife and PLOS. This is done due to the difference in structure and the average number of tokens in the lay summaries between the two datasets (i.e., $450$ and $800$ for PLOS and eLife, respectively).
Hyperparameters are set based on widely used values in the literature [@li2022clinicallongformer;@radford2019language;@Devlin2019BERTPO]. 

### Clinical Longformer Abstractor {#sec:clinical-longformer-abstractor}
The Clinical Longformer [@li2023comparative] is a transformer-based model that is pre-trained on the MIMIC-III dataset 
[@Johnson2016MIMICIII] and can process up to $4,096$ tokens in a single input sequence.
This is achieved by the implementation of a sparse attention mechanism that allows more computationally efficient processing of long-range dependencies.
We fine-tune the Clinical Longformer as a sequence-to-sequence task on pairs of (a) gold lay summaries and (b) ROUGE-maximising 
training data described in Section \ref{sec:extractor-network}. 

\begin{figure}
    \centering
    \includegraphics[width=0.5\textwidth]{charts/token_distribution}
    \caption{Token Distribution of Extracted Summaries}\label{fig:abstractor-eval-rouge}
\end{figure}

For the Longformer model, we experimented with window, batch, and input size to ensure that we would not run out of memory 
during training, as this is a common issue with such models [@orzhenovskii-2021-t5].
We found that a window size of $32$, batch size of $1$, and input size of $1,024$ worked best for our dataset, 
resulting in an evaluation loss of $3.4$ (Figure \ref{fig:abstractor-eval-loss}).

\begin{figure}
    \centering
    \includegraphics[width=0.49\textwidth]{charts/lf_eval_loss}
    \caption{Longformer evaluation loss}\label{fig:abstractor-eval-loss}
\end{figure}

### GPT-2 Abstractor {#sec:gpt2-abstractor}

The GPT-2 is an autoregressive language model that was trained using a casual language modeling objective [@radford_wu]. Given its extensive exposure to diverse text sources and natural language patterns, we hypothesize that GPT-2 would be particularly adept at generating lay summaries, making it a promising candidate for the abstractive summarization task. To fine-tune GPT-2 for this purpose, we utilize a "TL;DR" prompt, instructing the model to generate concise and informative summaries.

Similar to the Longformer, we train GPT-2 on both eLife and PLOS datasets, adopting most hyperparameters from the existing literature to ensure optimal performance [@bajaj-etal-2021-long]. Since GPT-2 can accommodate a total of 1024 tokens, we experimented with various splits between the number of tokens allocated for the extracted summary and the lay summary. Through experimentation, we determined that allocating $507$ tokens for the article and $512$ tokens for the summary, with $5$ reserved for the "TL;DR" prompt, yielded the best results in terms of summary quality and model performance. The evaluation loss decrease during the fine-tuning process is illustrated in Figure \ref{fig:gpt-eval}.

\begin{figure}
    \centering
    \includegraphics[width=0.49\textwidth]{charts/gpt_eval_loss}
    \caption{GPT 2 Evaluation Loss}\label{fig:gpt-eval}
\end{figure}

In the evaluation phase, we compared the performance of the GPT-2 Abstractor against the Clinical Longformer Abstractor, as well as other summarization models. The results indicate that both models have their strengths and weaknesses, which we will discuss in further detail in the following sections.

# Evaluation {#sec:evaluation}
In this section, we evaluate the performance of the summarization models described in Section \ref{sec:methods}.

# Quantitative Evaluation {#sec:evaluation-quantitative}

We compare our models by calculating the average F1 ROUGE scores on the PLOS evaluation dataset.
From Table \ref{tab:dataset_stats}, we can see that our Extractive Network performs as good as the standard ATS baseline - LexRank [@erkan2004] in terms of the lexical overlap with the gold lay summary.
On the other hand, we observe that the metrics decrease for the generative models due to their abstractive nature, 
which demonstrates how problematic and inconvenient for lay summarisation ROUGE is.
Nevertheless, it is clear that the Clinical Longformer outperforms considerably the GPT-2 perhaps due to the fact the latter is pre-trained on out-of-domain data.
Furthermore, we also note that there are insignificant differences in ROUGE between the top-10 and top-15 strategies of Sentence Extraction (see Section \ref{sec:extractor-network}) for the Clinical Longformer.

\begin{table}[htbp]
    \centering
    \begin{tabular}{|c|c|c|c|}
        \hline
        \textbf{Model} & \textbf{Rouge1} & \textbf{Rouge2} & \textbf{RougeL} \\
        \hline
            Lexrank & $0.34$ & $0.09$ & $0.16$ \\
        \hline
            Extractive & $0.33$ & $0.10$ & $0.16$ \\
        \hline
            GPT2 & $0.18$ & $0.02$ & $0.09$ \\
        \hline
            Longformer (top-15) & $0.28$ & $0.07$ & $0.15$ \\
        \hline
            Longformer (top-10) & $0.29$ & $0.06$ & $0.14$ \\
        \hline
    \end{tabular}
    \caption{ROUGE F1 Scores.}\label{tab:dataset_stats}
\end{table}

Regarding the readability of the generated summaries, it is clear and expected that our Extracted summary results in a 
low FKGL [@Kincaid1975DerivationON] and a high ARI [@senter1967automated] - meaning that it contains a lot of scientific jargon and is hard to read.
On the other hand, the GPT-2 and Longformer 

\begin{table}[htbp]
    \centering
    \begin{tabular}{|c|c|c|c|}
        \hline
        \textbf{Model} & \textbf{FKGL} & \textbf{ARI} & \textbf{Gunning} \\
        \hline
            Lay & 20.01 & 16.50 & 19.11 \\
        \hline
            Lex (Baseline) & 33.58 & 15.41 & 18.50 \\
        \hline
            Extractive & 10.60 & 25.01 & 26.22 \\
        \hline
            GPT2 & 30.68 & 21.36 & 23.26 \\
        \hline
            Longformer (top-15) & 23.84 & 19.62 & 20.62 \\
        \hline
            Longformer (top-10) & $27.33$ & $16.89$ & $18.44$ \\
        \hline
    \end{tabular}
    \caption{Readability metrics. \\ FKGL - higher is better, ARI and Gunning - lower is better}\label{tab:dataset_stats}
\end{table}


# Qualitative Evaluation {#sec:evaluation-qualitative}

# Discussion and Conclusion {#sec:discussion-conclusion}

## Limitations {#sec:limitations}
We identify the following limitations of our work:

1. **Readability Evaluation**: Although, we are evaluating our models with the traditional metrics: FKGL [@Kincaid1975DerivationON], ARI [@senter1967automated], and Gunning [@gunning1952technique],
   they are insufficient for the estimation of text readability in scientific writing. 
   Instead, what some researchers propose is to leverage masked language models [@martinc_readability] 
   like the noun-phrase BERT-based metric [@luo] that computes the probability of technical jargon.
   We appreciate that this method would have provided a more thorough evaluation of our models, and we leave it as future work.

2. **Limited input size**: Due to the limited available computational resources (i.e., Tesla V100-SXM2-16GB) we had to restrict 
   the input size of the Longformer to $1,024$ tokens (i.e., $4$ times less than the maximum size). Therefore, we could not make use of
   the full model capabilities in attending to long-range dependencies. This limitation propagates back to our extractor network,
   which produces only enough sentences to fit in the abstractor network. Thus, if we could increase the Longformer's input size, we could
   do the same for the Extractor model.

## Future Work {#sec:future-work}

In light of the limitations discussed, we propose multiple venues for future work:

1. **T5 Experimentation** [@clinicalt5]: We aim to develop and assess the Clinical T5 model as a specialized counterpart to the Clinical Longformer. The T5, a transformer-based model, boasts unique features like a denoising autoencoder in its pretraining objective, which is adept at reconstructing corrupted input text. This makes it suitable for our extractive approach, utilizing sentences from disparate article sections. 

2. **Clinical Longformer Enhancement** [@li2022clinicallongformer]: Our goal is to augment the Clinical Longformer's maximum token capacity by employing advanced hardware resources. This would facilitate experimentation with larger input dimensions and model training, potentially leading to superior summarization performance and more precise lay summaries.

3. **Feedback Integration**: We suggest incorporating readability and factual correctness rewards into our summarization pipeline using reinforcement learning methods [@scialom-etal-2019-answers]. 
   This can be achieved by the combination of the RNPTC metric [@luo] and the factual accuracy [@zhang-etal-2020-optimizing] into a single reward function, optimised via the Reinforce algorithm [@williams1992simple].  
   This approach aspires to promote the generation of summaries that are not only more comprehensible for non-experts but also more correct with respect to the input article.

## Conclusion {#sec:conclusion}

# Bibliography
