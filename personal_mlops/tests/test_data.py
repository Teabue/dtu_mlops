from torch.utils.data import Dataset
from personal_mlops.data import PCTreeDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = PCTreeDataset("data/raw")
    assert isinstance(dataset, Dataset), "Dataset is not an instance of torch.utils.data.Dataset."
    assert dataset[-1].shape[1] == 3, f"Data shape is incorrect, should be (N,3) go {dataset[-1].shape}."
    

