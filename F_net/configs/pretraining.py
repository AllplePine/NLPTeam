'''C4数据集的预训练的配置文件'''

from typing import Optional

from configs import base as base_config
from configs.base import ModelArchitecture, HybridAttentionLayout


def get_config():
    '''预训练的模型配置'''

    config = base_config.get_config()

    config.model_arch: ModelArchitecture = ModelArchitecture.F_NET

    config.mode: TrainingMode = TrainingMode.PRETRAINING

    config.train_batch_size: int = 64 #训练的批次大小

    config.eval_batch_size: int = 64 #评估的批次大小

    config.learning_rate: float = 1e-4 #学习率

    config.clipped_grad_norm: Optional[float] = None #梯度的裁剪

    config.num_train_steps: int = 1000000 #训练的步数

    config.num_warmup_steps: int = 10000 #评估的步数

    config.save_checkpoints_steps: int = 2000 #保存检查点的步数

    config.eval_frequency: int = 2000 #评估的频率

    config.max_num_eval_stewps: int = 100 #最大评估的步数

    config.init_checkpoint_dir: str = '' #初始化检查点的路径

    config.max_predicitons_per_seq: int = 80  #每个序列的最大预测

    config.masking_rate: float = 0.15 #掩码的概率

    config.mask_token_proportion: float = 0.8 #掩码的比例

    config.random_token_proportion: float = 0.1 #随机的比例

    config.trail: int = 0 #重复训练的虚拟参数

    return config