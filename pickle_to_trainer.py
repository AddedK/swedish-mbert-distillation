# This code converts a pickle file contained a distilled model into a fileformat that can be used by the Hugging Face trainers

import torch
import numpy as np
import transformers
import datasets
import pickle

path = "distilled_GIGA_FULL_100_MBERT_6_ADAPT_truncated\gs338429.pkl"
checkpoint = torch.load(path, map_location = torch.device('cpu'))

torch.save(checkpoint,"distilled_GIGA_FULL_100_MBERT_6_ADAPT_truncated.pth")
