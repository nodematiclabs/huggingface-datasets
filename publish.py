from datasets import Dataset, Features, Value
from huggingface_hub import HfApi, login

import os
import pandas as pd

data = {
    'text': [
        "This movie was fantastic!",
        "Worst experience ever.",
        "Pretty average film, nothing special.",
        "Absolutely loved it, would recommend!",
    ],
    'label': [1, 0, 0.5, 1]
}

df = pd.DataFrame(data)
# Do analysis, transformations, statistics, as needed

dataset = Dataset.from_pandas(df)

features = Features({
    'text': Value('string'),
    'label': Value('float'),
})

dataset = dataset.cast(features)

login(os.environ.get('HUGGINGFACE_API_KEY'))

dataset.push_to_hub(
    repo_id="nldemo/sentiment-demo",
    private=False,
)