import enum
import os.path as osp

__all__ = [
    "__abspath__",
    "Path",
    "Config",
]

class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()
    pass


__project_root__ = osp.dirname(osp.abspath(__file__))  # IMPROVE: use inspect.currentframe().f_back.f_back?

def __abspath__(*relative_paths_to_project_dir):
    return osp.join(__project_root__, *relative_paths_to_project_dir)

class Path:
    ExperimentsFolderAbs = __abspath__('experiments')
    DataFolderAbs = __abspath__('data')  # only for small dataset that resides inside of the project
    ModelsFolderAbs = __abspath__('models')
    DeployConfigAbs = __abspath__('config_deploy.json')

    @classproperty
    def ExperimentFolderAbs(self=None) -> str:
        return __abspath__(Path.ExperimentsFolderAbs, Config.ExperimentName)

    @classproperty
    def ExperimentConfigAbs(self=None) -> str:
        return __abspath__(Path.ExperimentsFolderAbs, Config.ExperimentName, 'config_experiment.json')

    @classproperty
    def ExperimentMainConfigAbs(self=None) -> str:
        return __abspath__(Path.ExperimentsFolderAbs, Config.ExperimentName, 'config_main.json')


class Config:
    class QuickTest:
        InputImagePath = "/tmp/ykk_waterdrop_black.jpg"  # SH06-0074_v002.jpg"  # for Test only
        GrayscaleImagePath = "/tmp/0_00001.jpg"
        ImagenetLabelsPath = "/tmp/Dataset/imagenet/imagenet_slim_labels.txt"

    __ExperimentNames__ = ['retrain/inceptresv2+scansnap(6class)',  # 2020/07/03
                           'tripletloss/inceptresv2_tlearn33c+tripletloss+ykk(5c,251)',  # 2020/05/25
                           'tripletloss/simple_sequential+tripletloss+mnist',  # 2020/04/09
                           '_test_/tf_1x_to_2x_3',  # 2020/03/31,筑基
                           'retrain/inceptionresnetv2+tlearn(33class)',  # 2020/03/26
                           '_test_/embedding_distance',  # 2020/03/24
                           'tripletloss/inceptresv2_tlearn33c+tripletloss+miniset(138+3,gray)'  # 2020/03/23
                           ]
    _ExperimentName = __ExperimentNames__[3]  # <== select an active experiment

    @classproperty  # IMPROVE: use class property
    def ExperimentName(self=None) -> str: return Config._ExperimentName

    @ExperimentName.setter
    def ExperimentName(self, value): Config._ExperimentName = value

    @ExperimentName.getter
    def ExperimentName(self): return Config._ExperimentName


# --- Obsolete Code ---------------------------------------------------------
# # ref: sklearn.utils :: deprecated
# @deprecated("--command_mode CLI argument is obsolete, use --experiment instead")
# class CommandMode(enum.Enum):
#     @classmethod
#     def hint_string(cls, sep=', '):
#         hints = []
#         for item in list(cls):
#             hints.append('{}: {}'.format(item.value, item.name))
#         return sep.join(hints)
#
#     Test = 0
#     RetrainModel = 1  # ref:model_retrain.py
#     UpdateFeatLib = 2
#     FeatComp = 3  # Model.Predict(X)=>y_ + Math.MinDistance(y_, y[])


# class Config:
#     """CLI argument:  parser.add_argument(
#         '--command_mode',
#         type=int,
#         default=Config.command_mode.value,
#         help=CommandMode.hint_string()
#     )
#     """
#     @deprecated("--command_mode CLI argument is obsolete, use --experiment instead")
#     @property
#     def command_mode(self): return CommandMode.Test
