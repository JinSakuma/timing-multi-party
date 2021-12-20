import sys
import os
from hydra.experimental import compose, initialize_config_dir


conf_dir = "/mnt/aoni04/jsakuma/development/timing-multi-party/conf"


class Config():
    """
    hydraによる設定値の取得 (conf)
    """
    @staticmethod
    def get_cnf(file_name="default.yaml"):
        """
        設定値の辞書を取得
        @return
            cnf: OmegaDict
        """
        if not os.path.isdir(conf_dir):
            print(f"Can not find file: {conf_dir}.")
            sys.exit(-1)

        with initialize_config_dir(config_dir=conf_dir):
            cnf = compose(config_name=file_name)
            return cnf

