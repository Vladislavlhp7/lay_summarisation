# pyright: reportUnusedImport=false, reportUnusedExpression=false

"""
# BioLaySum Demo App

Find out how to run the demo on CSF.
Hint: Check how you connect a port to CSF such as for Jupyter.
Now check if you can do the same for Streamlit.
"""

import streamlit as st
from laysummarisation.inference.extractor_model import perform_inference
from laysummarisation.model.extractor_model import load_extractor_model
from laysummarisation.process.greedy_rouge import process_entry
from laysummarisation.utils import load_jsonl_pandas, set_seed

# sys.path.append("../laysummarisation")


st.title("Abstractive Biomedical Lay Summarisation")

if st.checkbox("Show settings"):
    st.subheader("Settings")
    seed = st.number_input(
        "Seed", min_value=0, max_value=1000000, value=42, step=1
    )
    set_seed(int(seed))
    st.divider()


# TODO: Add a dropdown to select the dataset to use

datasets = ["eLife", "PLOS"]
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


@st.cache_resource
def load_extractor():
    return load_extractor_model("./weights/Bio_ClinicalBERT_2e-05/")


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

# Classification result
# Model input
# Model output


# Summarisation result
# Load the corresponning model
# Button to load the model so it doesn't load on every page load
# Model input
# Model output


# Rouge metric, comparison to model lay sum
