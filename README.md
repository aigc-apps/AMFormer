# AMFormer
The AMFormer algorithm, accepted at AAAI-2024, for deep tabular learning

### Quick Start
All configurations are written in the config file, and the training process can be started directly with the following command:
`python main.py --config config/run/ours_fttrans-hcdr.yaml`
### Configuration
Parameters are passed through `config/run/ours_<model_name>-<dataset_name>.yaml`. 
The config under run inherits the config of the model/data with the following format:
```python
__base__:
  - config/base_schedule.yaml
  - config/models/ours_fttrans_final.yaml
  - config/datasets/hcdr.yaml
```
### Input and output format
The model receives three inputs, which are:
Categorical feature data (shape = bs * num_cate), 
continuous feature data (shape = bs * num_cont), 
labels (shape = bs).

The output is a list: [logits, loss].

### Model's parameters
```
Default parameters start on line 278 of (models/AM_Former.py), with explanations starting on line 303:
- `dim`: The dimension of features, i.e., the `dim` in `bs * num_cate * dim`.
- `depth`: The number of layers of the transformer in the model.
- `heads`: The number of heads in the multi-head attention.
- `attn_dropout`: The dropout rate for the attention matrix.
- `ff_dropout`: The dropout rate for the feed-forward layers.
- `use_cls_token`: Whether to use a class token for classification (in practice, the difference is not significant).
- `groups`: The number of features in the next layer in AMFormer. Generally, if the number of features is small, this value can be equal to or larger than the input feature number; for a larger number of features (>200), feature reduction is performed, such as [128, 64, 64].
- `sum_num_per_group`: In each layer of AMFormer, the number of features each group needs to sum.
- `prod_num_per_group`: In each layer of AMFormer, the number of features each group needs for exponentiated multiplication (usually this value is less than or equal to sum_num_per_group).
- `cluster`: Whether to use AMFormer's attention; if `False`, vanilla attention is used.
- `target_mode`: The operation on the prompt, whether to mix with the original data.
- `token_descent`: If set to `True`, `cluster` must be used; if `False`, the use of `cluster` is optional.
- `use_prod`: Whether to use the multiplication module (which will increase computation).
- `num_special_tokens`: The number of special values in the input features, for example, 0 could represent a null value.
- `num_unique_categories`: The number of distinct categorical features.
- `out`: The number of classes for the output result.
- `num_cont`: The number of discrete feature columns (how many feature columns are discrete in the table).
- `num_cate`: The number of continuous feature columns (how many feature columns are continuous in the table).
- `use_sigmoid`: Whether to calculate sigmoid for the output when `out`=1.
```



