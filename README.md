# Fine-Grained-Requirement-Conflict-Detection-and-Resolution

This repository contains the code and dataset for our work on **fine-grained requirement conflict detection and resolution using element-level classification and multi-label large language models (LLMs)**.

Our approach models requirement conflicts at the element level (event, agent, operation, input, output, constraint), enabling multi-label conflict detection and structured resolution generation via LLMs.

For more details, please refer to our paper.

## Experimental Results

The results below cover **31 model–configuration combinations** across **7 models**, with three random seeds (**0, 42, and 2024**) per combination (93 runs in total). These metrics evaluate conflict detection; resolution-generation results are not included in the supplied data.

### Metrics and Reporting

- We report micro-precision, micro-recall, micro-F1, macro-F1, and weighted-F1. Higher values indicate better performance.
- Summary values are reported as **mean ± sample standard deviation** across the three seeds, on a **0–1 scale**, rounded to four decimal places.
- Each metric is averaged directly across runs; mean F1 is not recomputed from mean precision and mean recall.
- Configuration identifiers are retained from the experimental records. Only the supplied model–configuration combinations are listed.
- Zero-valued results are retained as reported and included in the summary statistics.

### Results Averaged Across Seeds

| Model | Configuration | Micro-Precision | Micro-Recall | Micro-F1 | Macro-F1 | Weighted-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| flan-t5-base | `lora_r8` | 0.6539 ± 0.5666 | 0.0990 ± 0.1591 | 0.1551 ± 0.2443 | 0.0402 ± 0.0617 | 0.1214 ± 0.1866 |
| flan-t5-base | `lora_r16` | 0.9491 ± 0.0278 | 0.2055 ± 0.1092 | 0.3278 ± 0.1397 | 0.0961 ± 0.0303 | 0.2714 ± 0.0890 |
| flan-t5-base | `lora_r32` | 0.8837 ± 0.0466 | 0.3943 ± 0.0335 | 0.5441 ± 0.0283 | 0.1906 ± 0.0384 | 0.4573 ± 0.0438 |
| flan-t5-base | `ia3_compact` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| flan-t5-base | `ia3_full` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Mistral-7B | `lora_r8` | 0.7766 ± 0.0059 | 0.6490 ± 0.0131 | 0.7070 ± 0.0070 | 0.4740 ± 0.0180 | 0.6405 ± 0.0068 |
| Mistral-7B | `lora_r16` | 0.8243 ± 0.0660 | 0.6463 ± 0.0338 | 0.7234 ± 0.0312 | 0.5131 ± 0.0159 | 0.6843 ± 0.0493 |
| Mistral-7B | `lora_r32` | 0.7860 ± 0.0652 | 0.6576 ± 0.0291 | 0.7146 ± 0.0252 | 0.5132 ± 0.0200 | 0.6632 ± 0.0289 |
| Mistral-7B | `ia3_compact` | 0.6450 ± 0.0188 | 0.5966 ± 0.0158 | 0.6198 ± 0.0161 | 0.4201 ± 0.0223 | 0.5894 ± 0.0303 |
| Mistral-7B | `ia3_full` | 0.6835 ± 0.0253 | 0.5998 ± 0.0107 | 0.6388 ± 0.0162 | 0.4327 ± 0.0134 | 0.6085 ± 0.0296 |
| Mistral-7B | `prompt_tokens20` | 0.6503 ± 0.0407 | 0.5559 ± 0.0231 | 0.5987 ± 0.0183 | 0.3932 ± 0.0498 | 0.5626 ± 0.0235 |
| Mistral-7B | `prompt_tokens40` | 0.6167 ± 0.0158 | 0.5361 ± 0.0363 | 0.5733 ± 0.0266 | 0.3442 ± 0.0442 | 0.5355 ± 0.0284 |
| Llama-3.2-1B-Instruct | `lora_r8` | 0.6839 ± 0.0345 | 0.5741 ± 0.0337 | 0.6241 ± 0.0326 | 0.4504 ± 0.0284 | 0.5603 ± 0.0450 |
| Llama-3.2-1B-Instruct | `lora_r16` | 0.7192 ± 0.0534 | 0.5639 ± 0.0094 | 0.6313 ± 0.0153 | 0.4507 ± 0.0136 | 0.5703 ± 0.0229 |
| Llama-3.2-1B-Instruct | `lora_r32` | 0.6775 ± 0.0237 | 0.5773 ± 0.0334 | 0.6233 ± 0.0288 | 0.4446 ± 0.0099 | 0.5513 ± 0.0413 |
| Llama-3.2-1B-Instruct | `ia3_compact` | 0.6171 ± 0.0486 | 0.4869 ± 0.0120 | 0.5436 ± 0.0181 | 0.3774 ± 0.0197 | 0.4478 ± 0.0182 |
| Llama-3.2-1B-Instruct | `ia3_full` | 0.6463 ± 0.0325 | 0.5152 ± 0.0181 | 0.5728 ± 0.0067 | 0.4011 ± 0.0209 | 0.4923 ± 0.0087 |
| Llama-3.2-1B-Instruct | `prompt_tokens20` | 0.6238 ± 0.0268 | 0.4446 ± 0.0335 | 0.5187 ± 0.0263 | 0.1932 ± 0.0548 | 0.4057 ± 0.0618 |
| Llama-3.2-1B-Instruct | `prompt_tokens40` | 0.6028 ± 0.0241 | 0.4473 ± 0.0786 | 0.5117 ± 0.0573 | 0.2375 ± 0.0922 | 0.4109 ± 0.1046 |
| Llama-3-8B | `lora_r32` | 0.8048 ± 0.0499 | 0.6886 ± 0.0265 | 0.7416 ± 0.0266 | 0.5571 ± 0.0186 | 0.7084 ± 0.0274 |
| Llama-2-13B | `lora_r8` | 0.7087 ± 0.0678 | 0.6094 ± 0.0481 | 0.6553 ± 0.0564 | 0.4690 ± 0.0523 | 0.5990 ± 0.0699 |
| Llama-2-13B | `lora_r32` | 0.7154 ± 0.0431 | 0.6019 ± 0.0475 | 0.6536 ± 0.0449 | 0.4464 ± 0.0558 | 0.5864 ± 0.0577 |
| Llama-2-13B | `ia3_compact` | 0.5813 ± 0.2139 | 0.4698 ± 0.1292 | 0.5186 ± 0.1652 | 0.3798 ± 0.0817 | 0.4888 ± 0.0972 |
| Llama-2-13B | `ia3_full` | 0.5830 ± 0.1661 | 0.5661 ± 0.0308 | 0.5664 ± 0.0780 | 0.4153 ± 0.0515 | 0.5602 ± 0.0226 |
| Llama-2-13B | `prompt_tokens40` | 0.6344 ± 0.0661 | 0.4045 ± 0.0305 | 0.4934 ± 0.0374 | 0.2448 ± 0.0940 | 0.3925 ± 0.0709 |
| Qwen1.5-32B | `lora_r8` | 0.7890 ± 0.0410 | 0.6110 ± 0.0304 | 0.6881 ± 0.0251 | 0.4663 ± 0.0131 | 0.6153 ± 0.0338 |
| Qwen1.5-32B | `lora_r16` | 0.7780 ± 0.0639 | 0.5998 ± 0.0234 | 0.6763 ± 0.0265 | 0.4546 ± 0.0213 | 0.6079 ± 0.0202 |
| Qwen1.5-32B | `lora_r32` | 0.8069 ± 0.0419 | 0.6281 ± 0.0213 | 0.7063 ± 0.0281 | 0.4989 ± 0.0103 | 0.6399 ± 0.0356 |
| Qwen1.5-32B | `ia3_compact` | 0.7624 ± 0.0450 | 0.5746 ± 0.0236 | 0.6543 ± 0.0045 | 0.3767 ± 0.0199 | 0.5843 ± 0.0065 |
| Qwen1.5-32B | `ia3_full` | 0.7873 ± 0.0264 | 0.5891 ± 0.0174 | 0.6736 ± 0.0094 | 0.4228 ± 0.0320 | 0.6075 ± 0.0130 |
| Yi-34B | `lora_r32` | 0.7920 ± 0.0462 | 0.6078 ± 0.0366 | 0.6866 ± 0.0211 | 0.4878 ± 0.0301 | 0.6214 ± 0.0328 |

Among the supplied configurations, **Llama-3-8B with `lora_r32`** achieves the highest mean micro-F1 (**0.7416 ± 0.0266**). Comparisons are descriptive; no statistical significance testing is reported.

### Results for Individual Seeds

The table below preserves every supplied run. Values are rounded to four decimal places for readability.

<details>
<summary>Expand all individual-seed results</summary>

| Model | Configuration | Seed | Micro-Precision | Micro-Recall | Micro-F1 | Macro-F1 | Weighted-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flan-t5-base | `lora_r8` | 0 | 0.9617 | 0.2825 | 0.4367 | 0.1113 | 0.3363 |
| flan-t5-base | `lora_r8` | 42 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flan-t5-base | `lora_r8` | 2024 | 1.0000 | 0.0144 | 0.0285 | 0.0092 | 0.0280 |
| flan-t5-base | `lora_r16` | 0 | 0.9193 | 0.3291 | 0.4846 | 0.1246 | 0.3669 |
| flan-t5-base | `lora_r16` | 42 | 0.9744 | 0.1220 | 0.2168 | 0.0643 | 0.1909 |
| flan-t5-base | `lora_r16` | 2024 | 0.9537 | 0.1653 | 0.2818 | 0.0993 | 0.2565 |
| flan-t5-base | `lora_r32` | 0 | 0.8859 | 0.4238 | 0.5733 | 0.2219 | 0.4886 |
| flan-t5-base | `lora_r32` | 42 | 0.9292 | 0.3579 | 0.5168 | 0.1478 | 0.4072 |
| flan-t5-base | `lora_r32` | 2024 | 0.8361 | 0.4013 | 0.5423 | 0.2022 | 0.4760 |
| flan-t5-base | `ia3_compact` | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flan-t5-base | `ia3_compact` | 42 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flan-t5-base | `ia3_compact` | 2024 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flan-t5-base | `ia3_full` | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flan-t5-base | `ia3_full` | 42 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flan-t5-base | `ia3_full` | 2024 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mistral-7B | `lora_r8` | 0 | 0.7806 | 0.6340 | 0.6997 | 0.4947 | 0.6432 |
| Mistral-7B | `lora_r8` | 42 | 0.7698 | 0.6549 | 0.7077 | 0.4632 | 0.6454 |
| Mistral-7B | `lora_r8` | 2024 | 0.7795 | 0.6581 | 0.7137 | 0.4640 | 0.6328 |
| Mistral-7B | `lora_r16` | 0 | 0.7776 | 0.6790 | 0.7249 | 0.5294 | 0.6962 |
| Mistral-7B | `lora_r16` | 42 | 0.7954 | 0.6116 | 0.6915 | 0.4978 | 0.6302 |
| Mistral-7B | `lora_r16` | 2024 | 0.8998 | 0.6485 | 0.7537 | 0.5122 | 0.7265 |
| Mistral-7B | `lora_r32` | 0 | 0.8135 | 0.6790 | 0.7402 | 0.5241 | 0.6956 |
| Mistral-7B | `lora_r32` | 42 | 0.8330 | 0.6244 | 0.7138 | 0.5254 | 0.6536 |
| Mistral-7B | `lora_r32` | 2024 | 0.7116 | 0.6693 | 0.6898 | 0.4902 | 0.6403 |
| Mistral-7B | `ia3_compact` | 0 | 0.6472 | 0.5859 | 0.6150 | 0.4214 | 0.5827 |
| Mistral-7B | `ia3_compact` | 42 | 0.6252 | 0.5891 | 0.6066 | 0.3971 | 0.5630 |
| Mistral-7B | `ia3_compact` | 2024 | 0.6626 | 0.6148 | 0.6378 | 0.4417 | 0.6225 |
| Mistral-7B | `ia3_full` | 0 | 0.6595 | 0.5875 | 0.6214 | 0.4324 | 0.5857 |
| Mistral-7B | `ia3_full` | 42 | 0.6811 | 0.6067 | 0.6418 | 0.4194 | 0.5978 |
| Mistral-7B | `ia3_full` | 2024 | 0.7100 | 0.6051 | 0.6534 | 0.4463 | 0.6419 |
| Mistral-7B | `prompt_tokens20` | 0 | 0.6587 | 0.5297 | 0.5872 | 0.3739 | 0.5456 |
| Mistral-7B | `prompt_tokens20` | 42 | 0.6061 | 0.5730 | 0.5891 | 0.3561 | 0.5527 |
| Mistral-7B | `prompt_tokens20` | 2024 | 0.6862 | 0.5650 | 0.6197 | 0.4498 | 0.5894 |
| Mistral-7B | `prompt_tokens40` | 0 | 0.6330 | 0.5538 | 0.5908 | 0.3939 | 0.5529 |
| Mistral-7B | `prompt_tokens40` | 42 | 0.6155 | 0.5602 | 0.5866 | 0.3096 | 0.5509 |
| Mistral-7B | `prompt_tokens40` | 2024 | 0.6016 | 0.4944 | 0.5427 | 0.3291 | 0.5028 |
| Llama-3.2-1B-Instruct | `lora_r8` | 0 | 0.7157 | 0.5859 | 0.6443 | 0.4530 | 0.5830 |
| Llama-3.2-1B-Instruct | `lora_r8` | 42 | 0.6473 | 0.5361 | 0.5865 | 0.4208 | 0.5085 |
| Llama-3.2-1B-Instruct | `lora_r8` | 2024 | 0.6888 | 0.6003 | 0.6415 | 0.4774 | 0.5895 |
| Llama-3.2-1B-Instruct | `lora_r16` | 0 | 0.7152 | 0.5602 | 0.6283 | 0.4592 | 0.5577 |
| Llama-3.2-1B-Instruct | `lora_r16` | 42 | 0.6679 | 0.5746 | 0.6178 | 0.4350 | 0.5565 |
| Llama-3.2-1B-Instruct | `lora_r16` | 2024 | 0.7746 | 0.5570 | 0.6480 | 0.4579 | 0.5967 |
| Llama-3.2-1B-Instruct | `lora_r32` | 0 | 0.6815 | 0.5666 | 0.6188 | 0.4465 | 0.5422 |
| Llama-3.2-1B-Instruct | `lora_r32` | 42 | 0.6521 | 0.5506 | 0.5970 | 0.4339 | 0.5153 |
| Llama-3.2-1B-Instruct | `lora_r32` | 2024 | 0.6989 | 0.6148 | 0.6541 | 0.4534 | 0.5964 |
| Llama-3.2-1B-Instruct | `ia3_compact` | 0 | 0.6718 | 0.4864 | 0.5642 | 0.3898 | 0.4681 |
| Llama-3.2-1B-Instruct | `ia3_compact` | 42 | 0.5791 | 0.4992 | 0.5362 | 0.3877 | 0.4426 |
| Llama-3.2-1B-Instruct | `ia3_compact` | 2024 | 0.6004 | 0.4751 | 0.5305 | 0.3546 | 0.4329 |
| Llama-3.2-1B-Instruct | `ia3_full` | 0 | 0.6789 | 0.5056 | 0.5796 | 0.4030 | 0.4988 |
| Llama-3.2-1B-Instruct | `ia3_full` | 42 | 0.6140 | 0.5361 | 0.5724 | 0.4211 | 0.4956 |
| Llama-3.2-1B-Instruct | `ia3_full` | 2024 | 0.6461 | 0.5040 | 0.5663 | 0.3793 | 0.4824 |
| Llama-3.2-1B-Instruct | `prompt_tokens20` | 0 | 0.6481 | 0.4286 | 0.5159 | 0.1989 | 0.4174 |
| Llama-3.2-1B-Instruct | `prompt_tokens20` | 42 | 0.5950 | 0.4222 | 0.4939 | 0.1358 | 0.3389 |
| Llama-3.2-1B-Instruct | `prompt_tokens20` | 2024 | 0.6284 | 0.4831 | 0.5463 | 0.2449 | 0.4607 |
| Llama-3.2-1B-Instruct | `prompt_tokens40` | 0 | 0.5774 | 0.4189 | 0.4856 | 0.2054 | 0.3790 |
| Llama-3.2-1B-Instruct | `prompt_tokens40` | 42 | 0.6055 | 0.3868 | 0.4721 | 0.1656 | 0.3260 |
| Llama-3.2-1B-Instruct | `prompt_tokens40` | 2024 | 0.6255 | 0.5361 | 0.5774 | 0.3415 | 0.5277 |
| Llama-3-8B | `lora_r32` | 0 | 0.8029 | 0.7191 | 0.7587 | 0.5726 | 0.7278 |
| Llama-3-8B | `lora_r32` | 42 | 0.7559 | 0.6709 | 0.7109 | 0.5365 | 0.6770 |
| Llama-3-8B | `lora_r32` | 2024 | 0.8557 | 0.6758 | 0.7552 | 0.5621 | 0.7204 |
| Llama-2-13B | `lora_r8` | 0 | 0.6745 | 0.5955 | 0.6326 | 0.4970 | 0.5811 |
| Llama-2-13B | `lora_r8` | 42 | 0.6648 | 0.5698 | 0.6137 | 0.4087 | 0.5398 |
| Llama-2-13B | `lora_r8` | 2024 | 0.7867 | 0.6629 | 0.7195 | 0.5014 | 0.6762 |
| Llama-2-13B | `lora_r32` | 0 | 0.6961 | 0.6067 | 0.6484 | 0.4485 | 0.5872 |
| Llama-2-13B | `lora_r32` | 42 | 0.6853 | 0.5522 | 0.6116 | 0.3896 | 0.5283 |
| Llama-2-13B | `lora_r32` | 2024 | 0.7647 | 0.6469 | 0.7009 | 0.5011 | 0.6437 |
| Llama-2-13B | `ia3_compact` | 0 | 0.7388 | 0.5538 | 0.6330 | 0.4226 | 0.5578 |
| Llama-2-13B | `ia3_compact` | 42 | 0.3378 | 0.3210 | 0.3292 | 0.2856 | 0.3776 |
| Llama-2-13B | `ia3_compact` | 2024 | 0.6673 | 0.5345 | 0.5936 | 0.4311 | 0.5310 |
| Llama-2-13B | `ia3_full` | 0 | 0.7689 | 0.5714 | 0.6556 | 0.4706 | 0.5862 |
| Llama-2-13B | `ia3_full` | 42 | 0.4490 | 0.5939 | 0.5114 | 0.3688 | 0.5477 |
| Llama-2-13B | `ia3_full` | 2024 | 0.5312 | 0.5329 | 0.5321 | 0.4064 | 0.5466 |
| Llama-2-13B | `prompt_tokens40` | 0 | 0.6547 | 0.4382 | 0.5250 | 0.3261 | 0.4603 |
| Llama-2-13B | `prompt_tokens40` | 42 | 0.5606 | 0.3788 | 0.4521 | 0.1419 | 0.3189 |
| Llama-2-13B | `prompt_tokens40` | 2024 | 0.6880 | 0.3965 | 0.5031 | 0.2664 | 0.3984 |
| Qwen1.5-32B | `lora_r8` | 0 | 0.8330 | 0.6244 | 0.7138 | 0.4745 | 0.6459 |
| Qwen1.5-32B | `lora_r8` | 42 | 0.7821 | 0.5762 | 0.6636 | 0.4512 | 0.5790 |
| Qwen1.5-32B | `lora_r8` | 2024 | 0.7519 | 0.6324 | 0.6870 | 0.4732 | 0.6209 |
| Qwen1.5-32B | `lora_r16` | 0 | 0.7955 | 0.6244 | 0.6996 | 0.4394 | 0.6306 |
| Qwen1.5-32B | `lora_r16` | 42 | 0.8314 | 0.5778 | 0.6818 | 0.4455 | 0.6013 |
| Qwen1.5-32B | `lora_r16` | 2024 | 0.7072 | 0.5971 | 0.6475 | 0.4789 | 0.5917 |
| Qwen1.5-32B | `lora_r32` | 0 | 0.8507 | 0.6404 | 0.7308 | 0.5095 | 0.6700 |
| Qwen1.5-32B | `lora_r32` | 42 | 0.8028 | 0.6404 | 0.7125 | 0.4889 | 0.6492 |
| Qwen1.5-32B | `lora_r32` | 2024 | 0.7673 | 0.6035 | 0.6757 | 0.4984 | 0.6006 |
| Qwen1.5-32B | `ia3_compact` | 0 | 0.7382 | 0.5795 | 0.6493 | 0.3914 | 0.5869 |
| Qwen1.5-32B | `ia3_compact` | 42 | 0.8143 | 0.5490 | 0.6558 | 0.3541 | 0.5770 |
| Qwen1.5-32B | `ia3_compact` | 2024 | 0.7347 | 0.5955 | 0.6578 | 0.3847 | 0.5892 |
| Qwen1.5-32B | `ia3_full` | 0 | 0.7899 | 0.6035 | 0.6843 | 0.4548 | 0.6211 |
| Qwen1.5-32B | `ia3_full` | 42 | 0.8124 | 0.5698 | 0.6698 | 0.3907 | 0.5953 |
| Qwen1.5-32B | `ia3_full` | 2024 | 0.7598 | 0.5939 | 0.6667 | 0.4230 | 0.6062 |
| Yi-34B | `lora_r32` | 0 | 0.7636 | 0.5859 | 0.6630 | 0.5003 | 0.5839 |
| Yi-34B | `lora_r32` | 42 | 0.7670 | 0.6501 | 0.7037 | 0.5096 | 0.6448 |
| Yi-34B | `lora_r32` | 2024 | 0.8453 | 0.5875 | 0.6932 | 0.4535 | 0.6354 |

</details>

Model names are standardized for display: `Llama-3___2-1B-Instruct` → `Llama-3.2-1B-Instruct`, `Qwen1___5-32B` → `Qwen1.5-32B`, `Llama-3-8b` → `Llama-3-8B`, and `Llama-2-13b` → `Llama-2-13B`.
