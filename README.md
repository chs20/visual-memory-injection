<h1 align="center"> Visual Memory Injection Attacks for Multi-Turn Conversations </h1>

<p align="center">
  Christian Schlarmann&nbsp;&nbsp;•&nbsp;&nbsp;Matthias Hein
  <br>
  <img src="assets/vmi.png" alt="Visual Memory Injection Attack" width="90%">
</p>


[[Paper]](assets/vmi.pdf) &nbsp; [[BibTeX]](#citation) 



<!-- <table width=90%">
  <tr>
    <td width="1%" align="left" style="white-space:nowrap;">
      <a href="">[Paper]</a>&nbsp;&nbsp;
      <a href="#citation">[BibTeX]</a>
    </td>
    <td align="center" style="white-space:nowrap;">
      Christian Schlarmann&nbsp;&nbsp;•&nbsp;&nbsp;Matthias Hein
    </td>
    <td width="1%"></td>
  </tr>
</table> -->


## Summary

<img src="assets/multi-turn-teaser.png" align="right" style="width: 40%; margin-left: 40px" alt="Teaser Figure" />

Generative large vision-language models (LVLMs) have recently achieved impressive performance gains, and their user base is growing rapidly. However, the security of LVLMs, in particular in a long-context multi-turn setting, is largely underexplored. In this repository, we consider the realistic scenario in which an attacker uploads a manipulated image to the web/social media. A benign user downloads this image and uses it as input to the LVLM. Our novel stealthy Visual Memory Injection (**VMI**) attack is designed such that on normal prompts the LVLM exhibits nominal behavior, but once the user gives a triggering prompt, the LVLM outputs a specific prescribed target message to manipulate the user, e.g. for adversarial marketing or political persuasion. Compared to previous work that focused on single-turn attacks, **VMI** is effective even after a long multi-turn conversation with the user. We demonstrate our attack on several recent open-weight LVLMs. We thereby show that large-scale manipulation of users is feasible with perturbed images in multi-turn conversation settings, calling for better robustness of LVLMs against these attacks.

**⚠ Note**: This repository releases adversarial attacks in order to support the development of more robust models and guardrails, and to raise awareness of the security risks of LVLMs. It is not intended nor allowed to be used for malicious purposes.

**Content:**
- [Summary](#summary)
- [Results](#results)
- [Installation](#installation)
- [Attack](#attack)
- [Inference](#inference)
  - [Paraphrases](#paraphrases)
  - [Transfer attacks](#transfer-attacks)
- [Evaluation](#evaluation)
- [Citation](#citation)



<br clear="right" />
<br clear="left" />

## Results
Below, we show success rates of the **VMI** attack on various models against the length of unrelated context at inference time, measured in number of tokens. Success is defined as the target message being reproduced given the corresponding prompt, while for unrelated context prompts the model behaves normally. We observe that the attack is effective even after long multi-turn conversations with the user. The attack remains effective after paraphrasing the anchor and target prompts, as well as when transferring the attack to a finetuned model, as shown in the paper. Several example conversation traces are supplied in [assets/conversations/](assets/conversations/).

<br>

<div align="center">
<img src="assets/sr_combined_comparison_n_tokens.png" style="width: 90%;" alt="Results" />
</div>



## Installation
Install python 3.12 and pytorch 2.4, then run 
```bash
pip install -r requirements.txt
```

## Attack
To run the **VMI** attack, first make sure the anchor targets for COCO are available. They are supplied in `clean_images/coco20/clean_outputs_{model_path}.json`. To regenerate them, run `python run_inference_coco_clean_anchor.py` (setting the model path inside the script). Also make sure the context outputs are available in `cache/{model_name}/{model_name}-diverse-v1-{config}.json`. To regenerate them, run `python run_inference_context_cache.py`.

Then set desired target strings, image dataset, and attack parameters in the `run_vmi.sh` script and run `bash run_vmi.sh`. The supplied configuration corresponds to the main attack runs in the paper.


## Inference
To run inference of the model given the adversarial images, run for example:
```bash
python run_multi_turn.py --model [Q3, Q2p5, llava-ov] --images landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear --prompts_name [diverse-v1 | holiday-v1 | diversetest-v1]
```

Replace `landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear` with the desired adversarial attack directory.

### Paraphrases
To run inference with paraphrased anchor and target prompts, set `--target_prompts [phone-alt3 | phone-alt4 | phone-alt5]`. Adapt "phone" to "car", "party", or "stock" for other target types. (In the paper we average over alt3, alt4, and alt5. alt1 and alt2 paraphrase only anchor OR target prompt, while alt3, alt4, and alt5 paraphrase both anchor and target prompt.)



### Transfer attacks
To transfer adversarial images generated against another model, set `--transfer-from-model [Q3, Q2p5, llava-ov]`.


## Evaluation
To evaluate the generated conversations, run
```bash
python -m evaluation.evaluate_messages --model [Q3, Q2p5, llava-ov]
```
To evaluate the transfer attacks, run
```bash
python -m evaluation.evaluate_messages --model [Q3, Q2p5, llava-ov] --transfer-from-model [Q3, Q2p5, llava-ov]
```

In order to plot the results, adapt and run `python run_token_count.py` to get the token counts for the conversations, and then run `python run_results_analysis.py` to obtain the plots.


## Citation
If you find this work useful, please consider citing our paper:
```bibtex
```