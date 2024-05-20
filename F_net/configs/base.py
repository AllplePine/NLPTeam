"""Base template config for pre-training and fine-tuning."""

import enum
import ml_collections

class ModelArchitecture(enum.Enum):
    """
    确定模型的架构，混合层
    """
    BERT = 'bert'
    F_NET = 'f_net' #傅里叶变换网络
    FF_ONLY = 'ff_only' #仅仅使用全连接层
    LINEAR = 'linear' #具有可学习权重的的矩阵乘法
    RANDOM = 'random' #随机初始化矩阵的权重

class TrainingMode(str, enum.Enum):
    """
    确定模型的训练模式 , 分为预训练和分类
    """
    PRETRAIN = 'pretraining'
    CLASSIFICATION = 'classification'

class HybridAttentionLayout(enum.Enum):
    """
    确定混合注意力层的布局，用注意力子层取代混合子层
    """
    BOTTOM = 'bottom'  # 第一个注意力子层
    MIDDLE = 'middle'  # 中间的注意力子层
    MIXED = 'mixed'  # 混合注意力子层
    TOP = 'top'  # 最后一个注意力子层

def get_config():
    '''
    预训练模型的基本的配置
    '''
    config = ml_collections.ConfigDict()

    #确定要使用的模型
    #特定的混合子层可以用注意力替换
    config.model_arch: ModelArchitecture = ModelArchitecture.F_NET

    config.save_checkpoints_steps: int = 1000  #保存检查点

    config.eval_frequency: int = 1000  #评估的频率

    config.learning_rate: float = 1e-4  #学习率

    config.init_checkpoint_dir: str = ''  #初始化检查点的路径

    config.do_lower_case: bool = True  #是否将输入转换为小写

    config.type_vocab_size: int = 4  #类型词汇表的大小

    config.d_emb: int = 768  #嵌入层的维度

    config.d_model: int = 768  #模型的维度

    config.d_ff: int = 3072  #前馈网络的维度

    config.max_seq_length: int = 512  #最大序列长度

    config.num_heads: int = 12  #头的数量

    config.num_layers: int = 12  #层数

    config.dropout_rate: float = 0.1  #丢失率

    config.mixing_dropout_rate: float = 0.1  #混合丢失率


    config.use_fft: bool = True  #是否使用傅里叶变换

    config.attention_layout: HybridAttentionLayout = HybridAttentionLayout.MIXED  #混合注意力层的布局

    config.num_attention_layers: int = 0  #注意力层的数量

    config.seed: int = 0  #种子

    config.trail:int = 0  #训练的轮数

    return config

