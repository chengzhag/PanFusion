from .PanoDataset import PanoDataset, PanoDataModule


class DemoDataset(PanoDataset):
    def load_split(self, mode):
        with open(self.data_dir) as f:
            data = f.readlines()
        data = [{'pano_prompt': d.strip()} for d in data]
        return data

    def get_data(self, idx):
        data = self.data[idx].copy()
        data['pano_id'] = f"{idx:06d}"
        return data


class Demo(PanoDataModule):
    def __init__(
            self,
            data_dir: str = 'data/demo.txt',
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset_cls = DemoDataset
