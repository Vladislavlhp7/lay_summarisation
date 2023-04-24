import os
import re
from random import seed
from typing import Dict, Tuple
from readability import Readability
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from nltk import sent_tokenize
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words
from transformers import BertTokenizerFast


def preprocess(text):
    """
    Preprocess a string of text.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = remove_full_stop_after_et_al(text)
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    regex = re.compile(r"\.([A-Z][a-z])")
    text = regex.sub(
        r". \1", text
    )  # Add space after full stop. Important for sentence tokenization
    return text


def sentence_tokenize(text):
    """
    Tokenize a string of text into sentences.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: A list of sentences.
    """
    return list(filter(lambda x: x.strip() != "", sent_tokenize(text)))


def lexrank_summarize(article: str, sentence_count: int = 25) -> str:
    """
    Use LexRank to generate a summary of an article.

    Args:
    article (str): The text of the article to summarize.
    sentence_count (int, optional): The number of sentences to include in the summary.
                                    Defaults to 25 due to 75% of the lay summaries being shorter than 25 sentences.

    Returns:
    str: The summary generated by LexRank.
    """

    # Initialize a parser and tokenizer for the article
    parser = PlaintextParser.from_string(article, Tokenizer("english"))

    # Initialize a LexRank summarizer with English stop words
    summarizer = LexRankSummarizer()
    summarizer.stop_words = get_stop_words("english")

    # Use LexRank to score the sentences and select the top K sentences
    summary_sentences = summarizer(parser.document, sentence_count)

    # Join the summary sentences into a single string and return it
    summary = " ".join(str(sentence) for sentence in summary_sentences)
    return summary


def process_data_to_model_inputs(
        batch, tokenizer, max_input_length, max_output_length, pre_summarise=True
):
    """
    Tokenize and preprocess a batch of data for use as model inputs.

    Args:
    batch (dict): A dictionary containing the input and output data for the batch.
    tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer object to use for tokenization.
    max_input_length (int): The maximum length of the input and output sequences after tokenization.
    max_output_length (int): The maximum length of the output sequences after tokenization.
    pre_summarise (bool): Whether to pre-summarise the input data before tokenization.

    Returns:
    dict: A dictionary containing the preprocessed model inputs for the batch.
    """

    if pre_summarise:
        # Use LexRank to summarize the articles in a batch
        article_summary = [lexrank_summarize(article) for article in batch["article"]]
    else:
        article_summary = batch["article"]

    # Tokenize the inputs and outputs using the provided tokenizer
    inputs = tokenizer(
        article_summary,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        batch["lay_summary"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    # Create a dictionary to store the preprocessed model inputs
    processed_batch = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }

    # Assign the tokenized inputs and attention masks to the processed batch dictionary

    # Create a list of 0s to use as the global attention mask
    global_attention_mask = [
        [0] * len(processed_batch["input_ids"][0])
        for _ in range(len(processed_batch["input_ids"]))
    ]
    # Set the first element of the global attention mask to 1 to indicate the start of the sequence
    global_attention_mask[0][0] = 1
    processed_batch["global_attention_mask"] = global_attention_mask

    # Assign the tokenized outputs and label masks to the processed batch dictionary
    processed_batch["labels"] = outputs.input_ids
    # Replace the PAD tokens with -100 to ignore them during training
    processed_batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in processed_batch["labels"]
    ]

    # Return the preprocessed model inputs as a dictionary
    return processed_batch


def load_article_dataset(fpath: str) -> Dataset:
    """
    Load an article dataset of a specified type from a given directory.

    Args:
    dtype (str): The type of dataset to load, such as 'train' or 'val'.
    filename (str): The name of the dataset file to load.
    directory (str): The directory path where the dataset file is located.

    Returns:
    Dataset: A Hugging Face Datasets object containing the loaded dataset.
    """

    # Load the dataset into a Pandas DataFrame
    df = pd.read_json(fpath, lines=True)

    # Convert the DataFrame to a Hugging Face Datasets object
    return Dataset.from_pandas(df)


def load_jsonl_pandas(fpath: str, nrows=None):
    """
    Load the the entire JSONL file into a Pandas DataFrame.

    Args:
    fpath (str): The path to the JSONL file to load.

    Returns:
    DataFrame: A Pandas DataFrame containing the first line of the JSONL file.
    """

    return pd.read_json(fpath, lines=True, nrows=nrows)


def create_article_dataset_dict(
        filename: str,
        batch_size: int,
        tokenizer,
        max_input_length: int,
        max_output_length: int,
        pre_summarise: bool = True,
) -> DatasetDict:
    """
    Create a dictionary of preprocessed datasets from article data in a given directory.

    Args:
        filename (str): The filename of the dataset to load.
        batch_size (int): The batch size to use for processing the dataset.
        tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer object to use for tokenization.
        max_input_length (int): The maximum length of the input and output sequences after tokenization.
        max_output_length (int): The maximum length of the output sequences after tokenization.
        pre_summarise (bool): Whether to pre-summarise the input data before tokenization.

    Returns:
        DatasetDict: A dictionary containing preprocessed datasets for training and validation.
    """

    # Define the dataset types to load
    dataset_types = ["train", "val"]

    # Initialize an empty dictionary to store the preprocessed datasets
    datasets = {}

    # Iterate through each dataset type and preprocess the data
    for dtype in dataset_types:
        # Load the dataset
        dataset = load_article_dataset(filename)

        # Preprocess the data for model inputs
        dataset = dataset.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "lay_summary", "headings"],
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_input_length": max_input_length,
                "max_output_length": max_output_length,
                "pre_summarise": pre_summarise,
            },
        )

        # Set the format of the dataset to be used with PyTorch
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
        )

        # Add the preprocessed dataset to the datasets dictionary
        datasets[dtype] = dataset

    # Return the preprocessed datasets as a DatasetDict
    return DatasetDict(datasets)


def set_seed(seed_v: int = 42) -> None:
    """
    Set the random seed for the random number generators used by NumPy, Python, and PyTorch.

    Args:
    seed_v (int): The value to use as the random seed.

    Returns:
    None
    """

    # Set the random seed for NumPy
    np.random.seed(seed_v)

    # Set the random seed for Python
    seed(seed_v)

    # Set the random seed for PyTorch on the CPU and GPU
    torch.manual_seed(seed_v)
    torch.cuda.manual_seed(seed_v)

    # Set the hash seed to a fixed value for consistent hash values
    os.environ["PYTHONHASHSEED"] = str(seed_v)

    # Print a message to indicate the random seed has been set
    print(f"Random seed set as {seed_v}")


def compute_metrics(pred, tokenizer) -> Dict[str, float]:
    """
    Compute Rouge2 Precision, Recall, and F-measure for given predictions and labels.

    Args:
        pred: A NamedTuple containing 'predictions' and 'label_ids' Tensors.
              'predictions' is a Tensor of predicted token IDs.
              'label_ids' is a Tensor of the ground truth token IDs.
        tokenizer: The tokenizer instance used for decoding the predictions and labels.

    Returns:
        A dictionary with Rouge2 Precision, Recall, and F-measure.
    """

    # Extract the label IDs and predicted IDs from the input NamedTuple
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Load the Rouge metric from the datasets library
    rouge = evaluate.load("rouge")

    # Decode the predicted and label IDs to strings, skipping special tokens
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Compute Rouge2 scores for the predictions and labels
    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )
    assert rouge_output is not None
    rouge_output = rouge_output["rouge2"].mid

    # Round the Rouge2 scores to 4 decimal places and return them in a dictionary
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def read_jsonl_data(path):
    """
    Load data from the given path.
    :param path: path to the data
    :return: data
    """
    return pd.read_json(path, lines=True)


def lexrank_data(data, max_length=130):
    """
    Repair the data using lexrank.

    :param data: data to be repaired
    :param max_length: max length of the summary
    :return: repaired data
    """
    summarizer = LexRankSummarizer()
    summaries = []
    for d in data:
        parser = PlaintextParser.from_string(d, Tokenizer("english"))
        summary = summarizer(parser.document, max_length)
        summaries.append(summary)
    return summaries


def get_binary_sentence_dataset(fname: str):
    """
    Load a binary sentence dataset from a CSV file.

    Args:
        fname (str): The filename of the dataset to load.

    Returns:
        Dataset: A Hugging Face Dataset object containing the loaded dataset.
    """
    df = pd.read_csv(fname)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: {"label": int(x["label"])})  # convert label to int
    dataset = dataset.class_encode_column("label")  # convert label to one-hot
    return dataset


def load_binary_data(
        fname: str, tokenizer: BertTokenizerFast, max_length: int = 128
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load a binary sentence dataset from a CSV file.

    Args:
        fname (str): The filename of the dataset to load.
        tokenizer (BertTokenizerFast): The tokenizer to use to tokenize the sentences.
        max_length (int): The maximum length of the tokenized sentences.

    Returns:
        Tuple(Dataset, Dataset, Dataset): A tuple containing the train, validation and test datasets.
    """
    # Get the dataset with the binary labels for each sentence in the article
    dataset = get_binary_sentence_dataset(fname)

    # Split the dataset into stratified train, validation and test (80%, 10%, 10%)
    dataset_split = dataset.train_test_split(
        test_size=0.2, seed=42, shuffle=True, stratify_by_column="label"
    )
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="label"
    )["train"]
    val_dataset = dataset_split["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="label"
    )["test"]

    # Tokenize the sentences
    def tokenize(batch):
        return tokenizer(
            batch["sentence"], padding=True, truncation=True, max_length=max_length
        )

    train_dataset = train_dataset.map(
        tokenize, batched=True, batch_size=len(train_dataset)
    )
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
    test_dataset = test_dataset.map(
        tokenize, batched=True, batch_size=len(test_dataset)
    )

    # Set the format to pytorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, val_dataset, test_dataset


def compute_binary_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    from pycm import ConfusionMatrix
    cm = ConfusionMatrix(actual_vector=labels, predict_vector=preds)

    tn, fp, fn, tp = cm.TN[0], cm.FP[0], cm.FN[0], cm.TP[0]
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def compute_readability_metrics_str(text: str):
    """
    Compute the readability metrics for the given text.

    Args:
        text (str): The text to compute the readability metrics for.

    Returns:
        Dict: A dictionary containing the computed readability metrics.
    """
    # Compute the readability metrics
    readability_metrics = Readability(text)

    # Return the readability metrics
    return {
        "flesch_kincaid_reading_score": readability_metrics.flesch().score,
        "ari_score": readability_metrics.ari().score,
        "gunning_fog_score": readability_metrics.gunning_fog().score,
    }


def compute_readability_metrics(summaries):
    """
    Compute the macro averaged readability metrics for the given summaries.

    Args:
        summaries (List[str]): The summaries to compute the readability metrics for.

    Returns:
        Dict: A dictionary containing the computed macro averaged readability metrics.
    """
    # Compute the readability metrics for each summary
    readability_metrics = [
        compute_readability_metrics_str(summary) for summary in summaries
    ]

    # Compute the macro averaged readability metrics
    macro_averaged_readability_metrics = {}
    for metric in readability_metrics[0].keys():
        macro_averaged_readability_metrics[metric] = np.mean(
            [readability_metric[metric] for readability_metric in readability_metrics]
        )

    # Return the macro averaged readability metrics
    return macro_averaged_readability_metrics


def remove_full_stop_after_et_al(text: str) -> str:
    return re.sub(r"(et al) \. (?![A-Z][a-z])", r"\1", text)


def main():
    lay_summary = preprocess("In the USA , more deaths happen in the winter than the summer . But when deaths occur varies greatly by sex , age , cause of death , and possibly region . Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment . Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed , or whether existing ones are effective . Scrutinizing seasonal patterns in death over time can also help scientists determine whether large-scale weather or climate changes are affecting the seasonality of death . Now , Parks et al . show that there are age and sex differences in which times of year most deaths occur . Parks et al . analyzed data on US deaths between 1980 and 2016 . While overall deaths in a year were highest in winter and lowest in summer , a greater number of young men died during summer \u2013 mainly due to injuries \u2013 than during winter . Seasonal differences in deaths among young children have largely disappeared and seasonal differences in the deaths of older children and young adults have become smaller . Deaths among women and men aged 45 or older peaked between December and February \u2013 largely caused by respiratory and heart diseases , or injuries . Deaths in this older age group were lowest during the summer months . Death patterns in older people changed little over time . No regional differences were found in seasonal death patterns , despite large climate variation across the USA . The analysis by Parks et al . suggests public health and medical interventions have been successful in reducing seasonal deaths among many groups . But more needs to be done to address seasonal differences in deaths among older adults . For example , by boosting flu vaccination rates , providing warnings about severe weather and better insulation for homes . Using technology like hands-free communication devices or home visits to help keep vulnerable elderly people connected during the winter months may also help .")
    article = preprocess("In temperate climates , winter deaths exceed summer ones . However , there is limited information on the timing and the relative magnitudes of maximum and minimum mortality , by local climate , age group , sex and medical cause of death . We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions . Death rates in men and women \u2265 45 years peaked in December to February and were lowest in June to August , driven by cardiorespiratory diseases and injuries . In these ages , percent difference in death rates between peak and minimum months did not vary across climate regions , nor changed from 1980 to 2016 . Under five years , seasonality of all-cause mortality largely disappeared after the 1990s . In adolescents and young adults , especially in males , death rates peaked in June/July and were lowest in December/January , driven by injury deaths . \n It is well-established that death rates vary throughout the year , and in temperate climates there tend to be more deaths in winter than in summer ( Campbell , 2017; Fowler et al . , 2015; Healy , 2003; McKee , 1989 ) . It has therefore been hypothesized that a warmer world may lower winter mortality in temperate climates ( Langford and Bentham , 1995; Martens , 1998 ) . In a large country like the USA , which possesses distinct climate regions , the seasonality of mortality may vary geographically , due to geographical variations in mortality , localized weather patterns , and regional differences in adaptation measures such as heating , air conditioning and healthcare ( Davis et al . , 2004; Braga et al . , 2001; Kalkstein , 2013; Medina-Ram\u00f3n and Schwartz , 2007 ) . The presence and extent of seasonal variation in mortality may also itself change over time ( Bobb et al . , 2014; Carson et al . , 2006; Seretakis et al . , 1997; Sheridan et al . , 2009 ) . A thorough understanding of the long-term dynamics of seasonality of mortality , and its geographical and demographic patterns , is needed to identify at-risk groups , plan responses at the present time as well as under changing climate conditions . Although mortality seasonality is well-established , there is limited information on how seasonality , including the timing of minimum and maximum mortality , varies by local climate and how these features have changed over time , especially in relation to age group , sex and medical cause of death ( Rau , 2004; Rau et al . , 2018 ) . In this paper , we comprehensively characterize the spatial and temporal patterns of all-cause and cause-specific mortality seasonality in the USA by sex and age group , through the application of wavelet analytical techniques , to over three decades of national mortality data . Wavelets have been used to study the dynamics of weather phenomena ( Moy et al . , 2002 ) and infectious diseases ( Grenfell et al . , 2001 ) . We also used centre of gravity analysis and circular statistics methods to understand the timing of maximum and minimum mortality . In addition , we identify how the percentage difference between death rates in maximum and minimum mortality months has changed over time . \n The strengths of our study are its innovative methods of characterizing seasonality of mortality dynamically over space and time , by age group and cause of death; using wavelet and centre of gravity analyses; using ERA-Interim data output to compare the association between seasonality of death rates and regional temperature . A limitation of our study is that we did not investigate seasonality of mortality by socioeconomic characteristics which may help with understanding its determinants and planning responses . \n We used wavelet and centre of gravity analyses , which allowed systematically identifying and characterizing seasonality of total and cause-specific mortality in the USA , and examining how seasonality has changed over time . We identified distinct seasonal patterns in relation to age and sex , including higher all-cause summer mortality in young men ( Feinstein , 2002; Rau et al . , 2018 ) . Importantly , we also showed that all-cause and cause-specific mortality seasonality is largely similar in terms of both timing and magnitude across diverse climatic regions with substantially different summer and winter temperatures . Insights of this kind would not have been possible analysing data averaged over time or nationally , or fixed to pre-specified frequencies . Prior studies have noted seasonality of mortality for all-cause mortality and for specific causes of death in the USA ( Feinstein , 2002; Kalkstein , 2013; Rau , 2004; Rau et al . , 2018; Rosenwaike , 1966; Seretakis et al . , 1997 ) . Few of these studies have done consistent national and subnational analyses , and none has done so over time , for a comprehensive set of age groups and causes of death , and in relation to regional temperature differences . Our results on strong seasonality of cardiorespiratory diseases deaths and weak seasonality of cancer deaths , restricted to older ages , are broadly consistent with these studies ( Feinstein , 2002; Rau et al . , 2018; Rosenwaike , 1966; Seretakis et al . , 1997 ) , which had limited analysis on how seasonality changes over time and geography ( Feinstein , 2002; Rau et al . , 2018; Rosenwaike , 1966 ) . Similarly , our results on seasonality of injury deaths are supported by a few prior studies ( Feinstein , 2002; Rau et al . , 2018; Rosenwaike , 1966 ) , but our subnational analysis over three decades revealed variations in when injury deaths peaked and in how seasonal differences in these deaths have changed over time in relation to age group which had not been reported before . A study of 36 cities in the USA , aggregated across age groups and over time , also found that excess mortality was not associated with seasonal temperature range ( Kinney et al . , 2015 ) . In contrast , a European study found that the difference between winter and summer mortality was lower in colder Nordic countries than in warmer southern European nations ( Healy , 2003; McKee , 1989 ) ( the study\u2019s measure of temperature was mean annual temperature which differed from the temperature difference between maximum and minimum mortality used in our analysis although the two measures are correlated ) . The absence of variation in the magnitude of mortality seasonality indicates that different regions in the USA are similarly adapted to temperature seasonality , whereas Nordic countries may have better environmental ( e . g . housing insulation and heating ) and health system measures to counter the effects of cold winters than those in southern Europe . If the observed absence of association between the magnitude of mortality seasonality and seasonal temperature difference across the climate regions also persists over time , the changes in temperature as a result of global climate change are unlikely to affect the winter-summer mortality difference . The cause-specific analysis showed that the substantial decline in seasonal mortality differences in adolescents and young adults was related to the diminishing seasonality of ( unintentional ) injuries , especially from road traffic crashes , which are more likely to occur in the summer months ( Liu et al . , 2005 ) and are more common in men . The weakening of seasonality in boys under five years of age was related to two phenomena: first , the seasonality of death from cardiorespiratory diseases declined , and second , the proportion of deaths from perinatal conditions , which exhibit limited seasonality ( Figure 9\u2014figure supplement 2 and Figure 10\u2014figure supplement 3 ) , increased ( MacDorman and Gregory , 2015 ) . In contrast to young and middle ages , mortality in older ages , where death rates are highest , maintained persistent seasonality over a period of three decades ( we note that although the percent seasonal difference in mortality has remained largely unchanged in these ages , the absolute difference in death rates between the peak and minimum months has declined because total mortality has a declining long-term trend ) . This finding demonstrates the need for environmental and health service interventions targeted towards this group irrespective of geography and local climate . Examples of such interventions include enhancing the availability of both environmental and medical protective factors , such as better insulation of homes , winter heating provision and flu vaccinations , for the vulnerable older population ( Katiyo et al . , 2017 ) . Social interventions , including regular visits to the isolated elderly during peak mortality periods to ensure that they are optimally prepared for adverse conditions , and responsive and high-quality emergency care , are also important to protect this vulnerable group ( Healy , 2003; Lerchl , 1998; Katiyo et al . , 2017 ) . Emergent new technologies , such as always-connected hands-free communications devices with the outside world , in-house cameras , and personal sensors also provide an opportunity to enhance care for the older , more vulnerable groups in the population , especially in winter when the elderly have fewer social interactions ( Morris , 2013 ) . Such interventions are important today , and will remain so as the population ages and climate change increases the within- and between-season weather variability . \n We used data on all 85 , 854 , 176 deaths in the USA from 1980 to 2016 from the National Center for Health Statistics ( NCHS ) . Age , sex , state of residence , month of death , and underlying cause of death were available for each record . The underlying cause of death was coded according to the international classification of diseases ( ICD ) system ( 9th revision of ICD from 1980 to 1998 and 10th revision of ICD thereafter ) . Yearly population counts were available from NCHS for 1990 to 2016 and from the US Census Bureau prior to 1990 ( Ingram et al . , 2003 ) . We calculated monthly population counts through linear interpolation , assigning each yearly count to July . We also subdivided the national data geographically into nine climate regions used by the National Oceanic and Atmospheric Administration ( Figure 18 and Table 2 ) ( Karl and Koss , 1984 ) . On average , the Southeast and South are the hottest climate regions with average annual temperatures of 18 . 4\u00b0C and 18\u00b0C respectively; the South also possesses the highest average maximum monthly temperature ( 27 . 9\u00b0C in July ) . The lowest variation in temperature throughout the year is that of the Southeast ( an average range of 17 . 5\u00b0C ) . The three coldest climate regions are West North Central , East North Central and the Northwest ( 7 . 6\u00b0C , 8 . 0\u00b0C , 8 . 2\u00b0C respectively ) . Mirroring the characteristics of the hottest climate regions , the largest variation in temperature throughout the year is that of the coldest region , West North Central ( an average range of 30 . 5\u00b0C ) , which also has the lowest average minimum monthly temperature ( \u22126 . 5\u00b0C in January ) . The other climate regions , Northeast , Southwest , and Central , possess similar average temperatures ( 10\u00b0C to 14\u00b0C ) and variation within the year of ( 23\u00b0C to 26\u00b0C ) , with the Northeast being the most populous region in the United States ( with 19 . 8% total population in 2016 ) . Data were divided by sex and age in the following 10 age groups: 0\u20134 , 5\u201314 , 15-24 , 25\u201334 , 35\u201344 , 45\u201354 , 55\u201364 , 65\u201374 , 75\u201384 , 85+\u00a0years . We calculated monthly death rates for each age and sex group , both nationally and for sub-national climate regions . Death rate calculations accounted for varying length of months , by multiplying each month\u2019s death count by a factor that would make it equivalent to a 31 day month . For analysis of seasonality by cause of death , we mapped each ICD-9 and ICD-10 codes to four main disease categories ( Table 1 ) and to a number of subcategories which are presented in the Supplementary Note . Cardiorespiratory diseases and cancers accounted for 56 . 4% and 21 . 2% of all deaths in the USA , respectively , in 1980 , and 40 . 3% and 22 . 4% , respectively , in 2016 . Deaths from cardiorespiratory diseases have been associated with cold and warm temperatures ( Basu , 2009; Basu and Samet , 2002; Bennett et al . , 2014; Braga et al . , 2002; Gasparrini et al . , 2015 ) . Injuries , which accounted for 8% of all deaths in the USA in 1980 and 7 . 3% in 2016 , may have seasonality that is distinct from so-called natural causes . We did not further divide other causes because the number of deaths could become too small to allow stable estimates when divided by age group , sex and climate region . We obtained data on temperature from ERA-Interim , which combines predictions from a physical model with ground-based and satellite measurements ( Dee et al . , 2011 ) . We used gridded four-times-daily estimates at a resolution of 80 km to generate monthly population-weighted temperature by climate region throughout the analysis period . We used wavelet analysis to investigate seasonality for each age-sex group . Wavelet analysis uncovers the presence , and frequency , of repeated maxima and minima in each age-sex-specific death rate time series ( Hubbard , 1998; Torrence and Compo , 1998 ) . In brief , a Morlet wavelet , described in detail elsewhere ( Cazelles et al . , 2008 ) , is equivalent to using a moving window on the death rate time series and analysing periodicity in each window using a short-form Fourier transform , hence generating a dynamic spectral analysis , which allows measuring dynamic seasonal patterns , in which the periodicity of death rates may disappear , emerge , or change over time . In addition to coefficients that measure the frequency of periodicity , wavelet analysis estimates the probability of whether the data are different from the null situation of random fluctuations that can be represented with white ( an independent random process ) or red ( autoregressive of order one process ) noise . For each age-sex group , we calculated the p-values of the presence of 12 month seasonality for the comparison of wavelet power spectra of the entire study period ( 1980\u20132016 ) with 100 simulations against a white noise spectrum , which represents random fluctuations . We used the R package WaveletComp ( version 1 . 0 ) for the wavelet analysis . Before analysis , we de-trended death rates using a polynomial regression , and rescaled each death rate time series so as to range between 1 and \u22121 . To identify the months of maximum and minimum death rates , we calculated the centre of gravity and the negative centre of gravity of monthly death rates . Centre of gravity was calculated as a weighted average of months of deaths , with each month weighted by its death rate; negative centre of gravity was also calculated as a weighted average of months of deaths , but with each month was weighted by the difference between its death rate and the year\u2019s maximum death rate . In taking the weighted average , we allowed December ( month 12 ) to neighbour January ( month 1 ) , representing each month by an angle subtended from 12 equally-spaced points around a unit circle . Using a technique called circular statistics , a mean ( \u03b8- ) of the angles ( \u03b81 , \u03b82 , \u03b83\u2026 , \u03b8n , ) representing the deaths ( with n the total number of deaths in an age-sex group for a particular cause of death ) is found using the relation below:\u03b8-=arg\u2211j=1nexp\u2061 ( i\u03b8j ) , where arg denotes the complex number argument and \u03b8j denotes the month of death in angular form for a particular death j . The outcome of this calculation is then converted back into a month value ( Fisher , 1995 ) . Along with each circular mean , a 95% confidence interval ( CI ) was calculated by using 1000 bootstrap samples . The R package CircStats ( version 0 . 2 . 4 ) was used for this analysis . For each age-sex group and cause of death , and for each year , we calculated the percent difference in death rates between the maximum and minimum mortality months . We fitted a linear regression to the time series of seasonal differences from 1980 to 2016 , and used the fitted trend line to estimate how much the percentage difference in death rates between the maximum and minimum mortality months had changed from 1980 to 2016 . We weighted seasonal difference by the inverse of the square of its standard error , which was calculated using a Poisson model to take population size of each age-sex group through time into account . This method gives us a p-value for the change in seasonal difference per year , which we used to calculate the seasonal difference at the start ( 1980 ) and end ( 2016 ) of the period of study . Our method of analysing seasonal differences avoids assuming that any specific month or group of months represent highest and lowest number of deaths for a particular cause of death , which is the approach taken by the traditional measure of Excess Winter Deaths . It also allows the maximum and minimum mortality months to vary by age group , sex and cause of death")
    print(compute_readability_metrics([lay_summary]))
    # print(compute_readability_metrics_str(article))

if __name__ == "__main__":
    main()
    pass
