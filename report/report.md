---
title: Lay It On Me. Generating Easy-to-Read Summaries for Non-Experts
author: Ahmed Soliman \And Marc Wenzlawski \And Vladislav Yotkov
header-includes: |
    \pdfoutput=1
    \usepackage{acl}
    \usepackage{natbib}
    \bibliographystyle{acl_natbib.bst}
---
\begin{abstract}
    In this work we design an extractive-abstractive lay summarisation pipeline for biomedical papers~\cite{biolaysumm-2023-overview} that generates summaries for non-experts. 
    For that purpose, we create a sentence-level ROUGE-maximising dataset from the gold summaries and the whole articles, which we then use to train a BERT-based classifier to identify the most important sentences per article. 
    Once an extracted summary is produced we feed it into three abstractive models (Clinical-T5~\cite{lehman2023clinical}, Clinical-Longformer~\cite{li2023comparative}, GPT-2~\cite{radford2019language}) that paraphrase the summary into a more readable version. 
    To evaluate our models we used the ROUGE metric~\cite{lin-2004-rouge} and the readability metrics - FKGL~\cite{Kincaid1975DerivationON}, Gunning Fog Score~\cite{gunning1952technique}, and ARI~\cite{senter1967automated} on the gold summaries and the generated summaries. 
    Results ...    
\end{abstract}
\section{Introduction}\label{sec:introduction}
\section{Related Work}\label{sec:related-work}
\section{Methods and Datasets}\label{sec:methods}
\section{Evaluation}\label{sec:evaluation}
\section{Discussion and Conclusion}\label{sec:discussion-conclusion}

\bibliography{custom}
