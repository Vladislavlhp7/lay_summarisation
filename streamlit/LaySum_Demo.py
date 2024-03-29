# pyright: reportUnusedImport=false, reportUnusedExpression=false

"""
# BioLaySum Demo App

Find out how to run the demo on CSF.
Hint: Check how you connect a port to CSF such as for Jupyter.
Now check if you can do the same for Streamlit.
"""

import pandas as pd
import torch
from rouge_score import rouge_scorer

import streamlit as st
from laysummarisation.inference.clinical_longformer import longformer_summary
from laysummarisation.inference.extractor_model import perform_inference
from laysummarisation.inference.gpt2 import gpt_summary
from laysummarisation.model.clinical_longformer import load_longformer_model
from laysummarisation.model.extractor_model import load_extractor_model
from laysummarisation.model.gpt2 import load_gpt_model
from laysummarisation.process.greedy_rouge import process_entry
from laysummarisation.utils import (
    compute_readability_metrics_str,
    load_jsonl_pandas,
    set_seed,
)

# sys.path.append("../laysummarisation")
st.markdown(
    """
    <style>
    .stTextArea [data-baseweb=base-input] {
        -webkit-text-fill-color: black;
    }

    .stTextArea [data-baseweb=base-input] [disabled=""]{
        -webkit-text-fill-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Abstractive Biomedical Lay Summarisation")

if st.checkbox("Show settings"):
    st.subheader("Settings")
    seed = st.number_input(
        "Seed", min_value=0, max_value=1000000, value=42, step=1
    )
    set_seed(int(seed))
    st.divider()


# TODO: Add a dropdown to select the dataset to use

datasets = ["PLOS", "eLife"]
corpus = st.selectbox("Select a corpus for the demo:", datasets)

st.write(f"Corpus: {corpus}")

data_load_state = st.text("Loading data...")

st.write("CHANGE THIS TO LOAD THE CORRECT DATASET (VAL)")


@st.cache_data
def load_dataset():
    return load_jsonl_pandas(f"./data/orig/val/{corpus}_val.jsonl", nrows=10)


df = load_dataset()

data_load_state.text("Done!")

if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(df.head(10))

# TODO: Add a select to select the datapoint to use

dp = st.slider("Select a datapoint:", 0, df.shape[0])

datapoint = df.iloc[dp]

# TODO: Add a section to show a preview of the datapoint


st.subheader("Datapoint preview")
if st.checkbox("Show datapoint preview"):
    st.text_area(
        f"Lay Summary ({len(datapoint['lay_summary'])} chars)",
        datapoint["lay_summary"],
        height=150,
        disabled=True,
    )
    st.text_area(
        f"Artice ({len(datapoint['article'])} chars)",
        datapoint["article"],
        height=150,
        disabled=True,
    )
    st.markdown("**Headings:** " + ", ".join(datapoint["headings"]))
    st.markdown("**Keywords:** " + ", ".join(datapoint["keywords"]))

# TODO: Add a section to show the pre-processing of the datapoint
# TODO: Maybe do a table with the nth top sentences by rouge, and their
# rouge scores, and whether they will be included in the final summary
# top n

# Rouge Maximisation
# Classification result
# Model input
# Model output

device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_extractor():
    return load_extractor_model("./weights/Bio_ClinicalBERT_2e-05/", device)


extractor_model, extractor_tokenizer = load_extractor()


@st.cache_data
def process(datapoint):
    return perform_inference(
        model=extractor_model,
        tokenizer=extractor_tokenizer,
        article=datapoint["article"],
        top_k=10,
    )


processed = process(datapoint)

# TODO: Preprocess with Vlad's model
st.subheader("Pre-processing: Rouge Maximisation")
if st.checkbox("Show pre-processing"):
    st.text_area(
        f"Processed Article ({len(processed)} chars)",
        processed,
        height=150,
        disabled=True,
    )


# Summarisation result
# GPT2
# Load the corresponning model
@st.cache_resource
def load_gpt():
    return load_gpt_model("./weights/gpt2/", device)


st.subheader("Summarisation: GPT2")
gpt_model, gpt_tokenizer = load_gpt()


@st.cache_data
def gpt_summarise(article):
    return gpt_summary(
        article=article,
        model=gpt_model,
        tokenizer=gpt_tokenizer,
        max_length=507,
    ).split(" TL;DR: ")[1]


gpt_final = gpt_summarise(processed)

if st.checkbox("Show summarisation GPT2"):
    if st.button("Refresh GPT"):
        gpt_final = gpt_summarise(processed)
    st.text_area(
        f"Summarisation ({len(gpt_final)} chars)",
        gpt_final,
        height=150,
        disabled=True,
    )


# Clinical Longformer
# Model input
# Model output
@st.cache_resource
def load_longformer():
    return load_longformer_model("./weights/longformer/", device)


st.subheader("Summarisation: Clinical Longformer")
longformer_model, longformer_tokenizer = load_longformer()


@st.cache_data
def longformer_summarise(article):
    return longformer_summary(
        article={"article": article},
        model=longformer_model,
        tokenizer=longformer_tokenizer,
        max_length=512,
    )["summary"][0]


longformer_final = longformer_summarise(processed)

if st.checkbox("Show summarisation Longformer"):
    if st.button("Refresh LF"):
        longformer_final = longformer_summarise(processed)
    st.text_area(
        f"Summarisation ({len(longformer_final)} chars)",
        longformer_final,
        height=150,
        disabled=True,
    )


def compute_metrics():
    return [
        compute_readability_metrics_str(s)
        if len(s.split(" ")) > 120
        else {
            "flesch_kincaid_reading_score": 0,
            "ari_score": 0,
            "gunning_fog_score": 0,
        }
        for s in [
            datapoint["lay_summary"],
            processed,
            gpt_final,
            longformer_final,
        ]
        # [datapoint["lay_summary"], gpt_final, longformer_final]
    ]


metrics = compute_metrics()

st.subheader("Evaluation: Readability Metrics")
if st.checkbox("Show metrics"):
    if len(gpt_final.split(" ")) < 120:
        st.warning(
            "GPT2 summary is too short to compute metrics, setting values to zero."
        )
    if len(longformer_final.split(" ")) < 120:
        st.warning(
            "Longformer summary is too short to compute metrics, setting values to zero."
        )

    st.table(
        pd.DataFrame(
            metrics, index=["Lay Summary", "Extractive", "GPT2", "Longformer"]
        )
    )
    st.write("flesch_kincaid_reading_score: higher is better")
    st.write("ari_score: lower is better")
    st.write("gunning_fog_score: lower is better")

rouge = rouge_scorer.RougeScorer(["rouge2", "rouge1", "rougeL"])

# Rouge metric, comparison to model lay sum
st.subheader("Evaluation: Rouge Metrics")
if st.checkbox("Show rouge metrics"):
    st.table(
        pd.DataFrame(
            [
                [
                    rouge.score(datapoint["lay_summary"], p)[r].fmeasure
                    for r in ["rouge1", "rouge2", "rougeL"]
                ]
                for p in [processed, gpt_final, longformer_final]
            ],
            index=["Extractive", "GPT2", "Longformer"],
            columns=["rouge1", "rouge2", "rougeL"],
        )
    )
