# swedish-mbert-distillation
The code and documentation for my Master's Thesis, which is about applying task-agnostic knowledge distillation on Swedish pre-trained mBERT models.

The report can be downloaded [here](https://kth.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=2&af=%5B%5D&searchType=UNDERGRADUATE&sortOrder2=title_sort_asc&language=en&pid=diva2%3A1698451&aq=%5B%5B%7B%22freeText%22%3A%22added+kina%22%7D%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-6142).

The azureML directory contains the files that were used together with Microsoft Azure Machine Learning, which includes the code for pre-training and task-agnostic distillation.

The SUCX_ft and wikiannEnFT notebooks are used to fine-tune the pre-trained teacher or student models on the respective datasets.

The code for evaluating the student models on the OverLim dataset can be found here: https://github.com/kb-labb/overlim_eval
