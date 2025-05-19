from datasets import load_dataset


class DataLoader:
    """Dataset Loader"""

    def __init__(
        self,
        test_size: float | None = 0.1,
        dataset_name=None
    ):
        self._test_size = test_size
        self._dataset_name = "datablations/c4-filter-small"
        if dataset_name:
            self._dataset_name = dataset_name

        self._dataset = load_dataset(self._dataset_name, split="train")

    def split_train_test(self):
        """Split train test set"""

        ds = self._dataset.select_columns(["text"])

        return ds.train_test_split(test_size=self._test_size)
