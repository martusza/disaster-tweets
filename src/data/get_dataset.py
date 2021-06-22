import os
from zipfile import ZipFile
import src.config as config
from kaggle.api.kaggle_api_extended import KaggleApi


class GetDataset:
    def __init__(self,
                 save_path: str = None,
                 kaggle_file_path: str = None):

        self.save_path = save_path
        self.kaggle_file_path = kaggle_file_path

    def get_dataset_from_kaggle(self,
                                kaggle_dataset_name: str,
                                unpack: bool = True):

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(kaggle_dataset_name,
                                       path=self.save_path)

        if unpack:
            self.unpack_file(zipfile_path=os.path.join(self.save_path, kaggle_dataset_name + '.zip'),
                             extract_path=self.save_path)

    def set_env_variable(self):
        os.environ['KAGGLE_CONFIG_DIR'] = self.kaggle_file_path

    @staticmethod
    def unpack_file(zipfile_path: str,
                    extract_path: str,
                    remove_pack: bool = True):

        zf = ZipFile(zipfile_path)
        zf.extractall(extract_path)
        zf.close()

        if remove_pack:
            os.remove(zipfile_path)


if __name__ == "__main__":
    GetDataset(save_path=config.DATASET_RAW).get_dataset_from_kaggle(kaggle_dataset_name='nlp-getting-started',
                                                                     unpack=True)
