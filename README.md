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

To access the internet from the CSF cluster, you need to use the `proxy` module:

```shell
module load tools/env/proxy2
```

Using the scripts in the `scripts` folder, you can automatically download and extract the data.

Run the following command to setup both data and model weights:

```shell
./scripts/setup_all.sh
```

Or individually using `./scripts/setup_data.sh` and `./scripts/setup_weights.sh`.

If downloading the data does not work, download it manually and extract the contents to `data/orig`

<https://drive.google.com/uc?id=1FFfa4fHlhEAyJZIM2Ue-AR6Noe9gOJOF&export=download>

For the weights, clone the following repo into the `weights` folder:

<https://huggingface.co/yikuan8/Clinical-Longformer>

## Running the code

First download the data and weights as described above.

Then run the following pre-processing code to pre-process the entire dataset:
```shell
./scripts/process_all.sh
```

Or run individually with `./scripts/data/elife.sh` and `./scripts/data/plos.sh`.

### CSF

Look at the `jobs` folder.

## Questions

How to use poetry on CSF?

How to setup pytorch for GPU on CSF?

Should we use MLFlow?

## Links

https://aclweb.org/aclwiki/BioNLP_Workshop
