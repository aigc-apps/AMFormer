### 启动训练
训练启动命令：
python main.py --config config/run/ours_fttrans-hcdr.yaml
### 配置文件
参数通过config/run/ours_fttrans-hcdr.yaml传递
run下的config通过以下内容来继承模型/数据的config
```python
__base__:
  - config/base_schedule.yaml
  - config/models/ours_fttrans_final.yaml
  - config/datasets/hcdr.yaml
```
### 输入/输出数据格式
dataset每次pop出来的数据格式为：
离散的特征数据（长度为num_cate的Tensor），连续的特征数据（长度为num_cont的Tensor），标签
输出给模型时会在batch size维度进行stack，最终得到`bs * num_cate`, `bs * num_cont`, `bs`的输入内容

输出为logit, loss
### 模型参数
```
默认参数在(models/AM_Former.py第278行开始，解释在303行开始)
dim 特征的维度，即bs * num_cate * dim里的dim
depth 模型中transformer的层数
heads 多头注意力里的头的个数
attn_dropout 注意力矩阵的dropout
ff_dropout 前馈层的dropout
use_cls_token 是否使用class token进行分类（实际上效果区别不大）
groups AMFormer中下一层特征数的数量。一般而言，在特征数量较少的情况下，如果该值=输入的特征树；特征数量较大时（>200）会进行特征减少，比如[128, 64, 64]
sum_num_per_group AMFormer每一层中每一组需要针对多少个特征进行求和
prod_num_per_group AMFormer每一层中每一组需要针对多少个特征进行求带幂乘法（一般该值小于等于sum_num_per_group）
cluster 是否使用AMFormer的注意力，如果为False就使用vanilla attention
target_mode 对prompt的操作，是否和原始数据进行混合
token_descent 如果设置为True，就必须使用cluster，如果是False，可以选择是否使用cluster
use_prod 是否使用乘法模块（会增加计算量）
num_special_tokens 输入特征中的特殊值比如0可能代表控制，一般默认为2就行
num_unique_categories 离散特征的数量
out 输出结果是几分类
num_cont 离散特征特征数（在表中有几列特征是离散的）
num_cate 连续特征特征数（在表中有几列特征是连续的）
use_sigmoid 在out=1的时候是否对输出结果计算sigmoid
```



