'''用于在 GLUE 和 SuperGLUE 基准上进行微调的配置。'''

from configs import base as base_config

from configs.base import ModelArchitecture, HybridAttentionLayout

def get_config():
    '''预训练的模型配置'''

    config = base_config.get_config()

    config.model_arch: ModelArchitecture = ModelArchitecture.F_NET

    config.mode: TrainingMode = TrainingMode.CLASSIFICATION

    config.dataset_name: str = 'glue/rte' #数据集的名称

    config.save_checkpoints_steps: int = 200 #保存检查点的步数

    config.eval_proportion: float = 0.05 #评估的比例

    config.train_batch_size: int = 64 #训练的批次大小

    config.eval_batch_size: int = 32 #评估的批次大小

    config.learning_rate: float = 1e-5 #学习率

    config.num_train_epochs: int = 3 #训练的轮数

    config.warmup_proportion: float = 0.1 #热身的比例

    config.max_num_eval_steps: int = 10000 #最大评估的步数

    config.init_checkpoint_dir: str = '' #初始化检查点的路径

    config.trail: int = 0 #重复训练的虚拟参数
    
    return config