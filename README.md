# laySummarisation

Project as part of COMP34812: Natural Language Understanding

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

Run the following command to setup both data:

```shell
./scripts/setup_data.sh
```

If downloading the data does not work, download it manually and extract the contents to `data/orig`

<https://drive.google.com/uc?id=1FFfa4fHlhEAyJZIM2Ue-AR6Noe9gOJOF&export=download>

## Running the code

First download the data and weights as described above.

Then run the following pre-processing code to pre-process the entire dataset:

```shell
./scripts/process_all.sh
```

Or run individually with `./scripts/data/elife.sh` and `./scripts/data/plos.sh`.

### CSF

Look at the `jobs` folder.

## Models

Extractor model (based on <https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT>):
<https://drive.google.com/drive/folders/1w-LhpA1ek5V10wUImU0BiRhjNOjihZiQ?usp=sharing>

GPT2 Model (based on <https://huggingface.co/gpt2>):
<https://livemanchesterac-my.sharepoint.com/:f:/g/personal/ahmed_soliman-2_student_manchester_ac_uk/EjcG0NcNbpRKoCvUUzzRCrwBKNE4RN-IQfh_ZUiVB3Tkvg?e=SOGPs1>

Clinical Longformer model (based on <https://huggingface.co/yikuan8/Clinical-Longformer>):
<https://drive.google.com/drive/folders/1QtFVqKHtmj_T64Vanyyrm5mnigl7_TJ4?usp=sharing>

ClinicalT5 model (based on <https://physionet.org/content/clinical-t5/1.0.0/>):
<https://livemanchesterac-my.sharepoint.com/:f:/g/personal/ahmed_soliman-2_student_manchester_ac_uk/EjcG0NcNbpRKoCvUUzzRCrwBKNE4RN-IQfh_ZUiVB3Tkvg?e=SOGPs1>

## Dataset

The datasets used was provided as part of the BioLaySum 2023 Challenge. The two datasets are academic articles from PLOS and eLife.

The data can be accessed at the following link:
<https://biolaysumm.org/>

(If that link doesn't work check the google drive link above)

Data source:

[1] Tomas Goldsack, Zhihao Zhang, Chenghua Lin, Carolina Scarton. Making Science Simple: Corpora for the Lay Summarisation of Scientific Literature. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022), Abu Dhabi. url

[2] Zheheng Luo, Qianqian Xie, Sophia Ananiadou. Readability Controllable Biomedical Document Summarization. Findings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022 Findings), Abu Dhabi. url

## Links

https://aclweb.org/aclwiki/BioNLP_Workshop
