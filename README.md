# laySummarisation

Project as part of COMP34812: Natural Language Understanding

# TODO


General Theme: Attempt to find a way to fit the necessary contextual information into the limit context size for the T5 model. 

## General

- Multi-gpu training
- Sharing the tuned models
- Evaluation
- Code for training and evaluation
- Baseline???
- Metrics (is rouge enough for lay sum?)
- tuning on elife + on plos or separately?
- Optimum parameters for LexRank
  - Number of sentences
  - Have to consider max token length of 4096 (input length + max output length = 4096)
  - How will it impact traning time?

## Vlad

- 

## Ahmed

- 

## Marc

- 

## Installation

Preferably use poetry. See install info [here](https://python-poetry.org/docs/).

When using it with VSCode, or any IDE with local venv management, you should include the venv created by poetry in your local directory. To achieve this use the following command before installing:

```shell
poetry config virtualenvs.in-project true
```

Then use the following command to install all dependencies:

```shell
potetry install
```

For further info on managing packages etc, see the Poetry docs.

## Data and weights

Using the scripts in the `scripts` folder, you can automatically download and extract the data.

Before you download the model weights, ensure you crate a `.env` file with the following contents:

```shell
USERNAME=<your username>
PASSWORD=<your password>
```

Then just run the following commands:

```shell
chmod +x scripts/*
./scripts/prep_all.sh
```

If the file download does not work, you can download the dev data manually from the following link:

<https://drive.google.com/uc?id=1FFfa4fHlhEAyJZIM2Ue-AR6Noe9gOJOF&export=download>

## Running the code

### Locally

Don't

### CSF

Look at the `jobs` folder.

## Questions

How to use poetry on CSF?

How to setup pytorch for GPU on CSF?

Should we use MLFlow?

## Links

https://aclweb.org/aclwiki/BioNLP_Workshop
