import itertools
import json
import os

import matplotlib.pyplot as plt#
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from tueplots import bundles, axes, cycler, figsizes
from tueplots.constants.color import palettes
from tueplots.constants import markers
from tueplots.constants.color import rgb
from utils.naming import ATTACK_SHORT_NAMES, DESIGN_ABLATION_RUNS, ITERATION_ABLATION_RUNS, MAIN_PAPER_RUNS, TRANSFER_RUNS


TARGET_NAME_TO_STRING = {
    "motorola": r"I strongly recommend\\the Motorola Moto G56.",
    "car": r"I strongly recommend\\the Apple iCar.",
    "party": r"You should vote for the\\National Freedom Party (NFP).",
    "stock": r"You should buy GameStop (GME) stock immediately."
}


def get_short_label(res_dir, ablation=None):
    # assert res_dir.split("/")[-1] in MAIN_PAPER_RUNS, f"Not main-paper run: {res_dir}"
    if res_dir == "Qwen-3-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6-cycle5-linear":
        if not ablation:
            return {"label": "Qwen3-VL", "color": rgb.tue_dark, "marker": "o"}
        elif ablation == "design":
            return {"label": r"Qwen3-VL (\textbf{\textsc{vmi}})", "color": rgb.tue_dark, "marker": "o"}
        elif ablation == "iterations":
            return {"label": "Qwen3-VL (2k)", "color": rgb.tue_dark, "marker": "o"}
    elif res_dir == "Qwen-3-8B/landmarks-attack2000-eps8-stock-insert-diverse-v1-6":
        return {"label": "Qwen3-VL (w/o cycle)", "color": rgb.tue_brown, "marker": "d"}
    elif res_dir == "Qwen-3-8B/landmarks-attack2000-eps8-stock":
        return {"label": r"Qwen3-VL (w/o cycle \& ctx)", "color": rgb.tue_mauve, "marker": "^"}
    elif res_dir == "Qwen-3-8B/landmarks-attack2000-eps8-stock-single-target":
        return {"label": "Qwen3-VL (single target)", "color": rgb.tue_gray, "marker": "p"}
    elif res_dir == "Qwen-3-8B/landmarks-attack8000-eps8-stock-insert-diverse-v1-6-cycle5-linear":
        return {"label": "Qwen3-VL (8k)", "color": rgb.tue_lightblue, "marker": "d"}
    elif res_dir == "Qwen-3-8B/landmarks-attack500-eps8-stock-insert-diverse-v1-6-cycle5-linear":
        return {"label": "Qwen3-VL (500)", "color": rgb.tue_darkgreen, "marker": "^"}
    elif res_dir.startswith("Qwen-2p5-7B/"):
        return {"label": "Qwen2.5-VL", "color": rgb.tue_red, "marker": "d"}
    elif res_dir.startswith("Qwen-3-8B/"):
        return {"label": "Qwen3-VL", "color": rgb.tue_dark, "marker": "o"}
    elif res_dir.startswith("LLaVA-OV-1.5-8B/"):
        return {"label": "LLaVA-OV-1.5", "color": rgb.tue_gold, "marker": "^"}
    elif res_dir.startswith("Qwen-3-8B_TO_Q3-sealionv4-8B/"):
        return {"label": r"Qwen3-VL$\to$SEA-LION", "color": rgb.tue_violet, "marker": "s"}
    elif res_dir.startswith("Qwen-3-8B_TO_Q3-med3-8B/"):
        return {"label": r"Qwen3-VL$\to$Med3", "color": rgb.tue_blue, "marker": "x"}
    else:
        raise ValueError(f"Unknown model: {res_dir}")


def plot_token_vs_score_bins(results, bins=25, logx=True):
    # assert len(set(results[]["conversation_type"] for k in results)) == 1, "All conversation types must be the same"
    # assert len(set(results[k]["target_type"] for k in results)) == 1, "All target types must be the same"
    # conversation_type = results[list(results.keys())[0]]["conversation_type"]

    fig, axs = plt.subplots(nrows=len(results), ncols=len(results[list(results.keys())[0]]), sharey=True, sharex=True)
    # increase figure size
    fig.set_size_inches(15, 10)
    for i, target_type in enumerate(results):
        for j, conversation_type in enumerate(results[target_type]):
            for attack_dir in results[target_type][conversation_type]:
                ax = axs[i, j]
                df = pd.DataFrame(results[target_type][conversation_type][attack_dir]).sort_values("n_tokens")
                # df["bin"] = pd.cut(df["n_tokens"], bins=bins)  # equal width bins
                df["bin"] = pd.qcut(df["n_tokens"], q=bins, duplicates="raise") # equal number of samples per bin
                g = df.groupby("bin", observed=True)["combined_score"]
                mean = g.mean()
                mid = mean.index.map(lambda b: 0.5 * (b.left + b.right)).to_numpy()

                mean = mean.to_numpy()
                n = g.size()
                se = np.sqrt(mean * (1 - mean) / n)

                label = ATTACK_SHORT_NAMES[attack_dir.replace("logs/", "")]
                ax.plot(mid, mean, "o-", label=label)
                ax.fill_between(mid, mean - se, mean + se, alpha=0.2)
                if logx:
                    ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{conversation_type} - {target_type}")
            ax.grid(visible=True, which="major", axis="both")
            if j == 0:
                ax.set_ylabel("Combined success rate")
                ax.legend()
        # if i == 1:
        #     ax.set_title(f"{conversation_type} - {target_type}")
        if i == len(results) - 1:
            ax.set_xlabel("Number of tokens")
    plt.tight_layout()
    plt.savefig("output/token_vs_score.png", dpi=200)
    plt.close()


def plot_results(results, x_data_name="n_tokens", logx=False, include_title=True, ablation=None):
    """
    Plot the average number of tokens after each conversation turn per model against the combined success rate.
    """
    n_rows = len(results)
    n_cols = len(results[list(results.keys())[0]])
    x_data_name_to_string = {
        "n_tokens": "Tokens",
        "n_turns": "Conversation Turns",
    }

    # set up tueplots
    bundle = bundles.icml2024()
    plt.rcParams.update(bundle)
    plt.rcParams.update(figsizes.icml2024_full(nrows=n_rows, ncols=n_cols, height_to_width_ratio=0.55))
    plt.rcParams.update(axes.lines())
    # plt.rcParams.update(cycler.cycler(color=palettes.tue_plot[:3], marker=markers.o_sized[[0, 2, 3]]))
    plt.rcParams.update(cycler.cycler(color=palettes.tue_plot[:5], marker=['d', 'o', '^', 'p', '+']))
    plt.rcParams['lines.markersize'] = 2.5
    plt.rcParams.update({"figure.dpi": 200})  # increase fig size in notebook
    # # increase figure size
    # fig.set_size_inches(15, 10)

    # legend_loc = "upper right"
    legend_loc = "lower left"
    ylabel_offset = 0.42 if n_rows == 1 else 0.49 # y<0.5 moves label down

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharey=True, sharex=True)
    fig.set_constrained_layout_pads(
        wspace=0.02, # horizontal space between subplots
        hspace=0.01, # vertical space between subplots
        w_pad=0.0, # horizontal padding between subplots and figure edges
        h_pad=0.0, # vertical padding between subplots and figure edges
    )


    # make sure axs is a 2D array
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)
    for i, target_type in enumerate(results):
        for j, conversation_type in enumerate(results[target_type]):
            for attack_dir in results[target_type][conversation_type]:
                ax = axs[i, j]
                if x_data_name == "n_tokens":
                    x = results[target_type][conversation_type][attack_dir]["n_tokens"]
                elif x_data_name == "n_turns":
                    x = results[target_type][conversation_type][attack_dir]["n_turns"]
                else:
                    raise ValueError(f"Invalid x_data_name: {x_data_name}")
                scores = results[target_type][conversation_type][attack_dir]["sr_combined"]
                scores_std = results[target_type][conversation_type][attack_dir].get("sr_combined_std", None)
                if False:  # longer label
                    label = ATTACK_SHORT_NAMES[attack_dir.replace("logs/", "")]
                else:
                    label_dict = get_short_label(attack_dir.replace("logs/", ""), ablation=ablation)
                    label_str = label_dict["label"]
                    color = label_dict["color"]
                    marker = label_dict["marker"]
                    if ablation == "design":
                        label_str = label_str.replace("Qwen3-VL (", "").replace(")", "")
                # sort by x data
                sorting_mask = np.argsort(x)
                x = np.array(x)[sorting_mask]
                scores = np.array(scores)[sorting_mask]
                if scores_std is not None:
                    scores_std = np.array(scores_std)[sorting_mask]
                # now plot
                ax.plot(x, scores, label=label_str, color=color, marker=marker)
                # plot std-dev as shaded region if available
                if scores_std is not None:
                    ax.fill_between(x, scores - scores_std, scores + scores_std, color=color, alpha=0.2)
                if logx:
                    ax.set_xscale("log")
            # formatting
            ax.set_ylim(0, 105)
            # ax.set_xlim(0, 14000)  # set this for transfer runs, other xaxis to short
            ax.yaxis.set_major_formatter(PercentFormatter(100))
            ax.grid(visible=True, which="major", axis="both")
            if j == 0: # first col
                target_string = TARGET_NAME_TO_STRING[target_type]
                y_label = r"\parbox{3cm}{\textbf{Target:} \textit{" + target_string + r"}"
                ax.set_ylabel(y_label, fontsize=6, y=ylabel_offset)  # y<0.5 moves label down
                # ax.set_ylabel("Combined success rate")
                ax.legend(loc=legend_loc) if ablation != "design" else None
            else: # not first col
                ax.tick_params(axis='y', which='both', length=0)
            if i == 0: # first row
                if include_title:
                    ax.set_title(f"{conversation_type}")
            if i == len(results) - 1: # last row
                if j == 1: # second col
                    pass
                    # ax.set_xlabel(x_data_name_to_string[x_data_name])
            else: # not last row
                ax.tick_params(axis='x', which='both', length=0)
            
    # trim legend
    if ablation == "design":
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, 
            loc="upper right", ncol=1, 
            # fontsize=6,
            # bbox_to_anchor=(0.66, 0.98),  # top right of middle plot
            bbox_to_anchor=(0.384, 1.02),  # top right of first plot
            columnspacing=0.25, labelspacing=0.1,
            handletextpad=0.3,
            handlelength=0.9,
            borderpad=0.2,
        )
    
    # create fname
    paraphrases_str = "_paraphrases" if paraphrases else ""
    if not ablation:
        fname = f"sr_combined{paraphrases_str}_comparison_{x_data_name}"
    elif ablation == "design":
        fname = f"sr_combined_algorithm-ablation_comparison_{x_data_name}"
    elif ablation == "iterations":
        fname = f"sr_combined_iterations-ablation_comparison_{x_data_name}"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    if attack_dirs == TRANSFER_RUNS:
        fname = fname + "_transfer"
    # save figure
    plt.savefig(f"output/{fname}.pdf")
    # plt.savefig(f"output/{fname}.png", dpi=200)
    print(f"Saved figure to output/{fname}.pdf")
    # also save to out.pdf
    plt.savefig(f"output/out.pdf")
    plt.close()

def plot_results_single_column(results, x_data_name="n_tokens", logx=False, include_title=True, ablation="design"):
    """
    Plot the average number of tokens after each conversation turn per model against the combined success rate.
    """
    n_rows = 1
    n_cols = 2
    x_data_name_to_string = {
        "n_tokens": "Tokens",
        "n_turns": "Conversation Turns",
    }

    # set up tueplots
    bundle = bundles.icml2024()
    plt.rcParams.update(bundle)
    plt.rcParams.update(figsizes.icml2024_half(nrows=n_rows, ncols=n_cols))
    plt.rcParams.update(axes.lines())
    # plt.rcParams.update(cycler.cycler(color=palettes.tue_plot[:3], marker=markers.o_sized[[0, 2, 3]]))
    plt.rcParams.update(cycler.cycler(color=palettes.tue_plot[:5], marker=['d', 'o', '^', 'p', '+']))
    plt.rcParams['lines.markersize'] = 1.75
    plt.rcParams['lines.linewidth'] = 0.75
    plt.rcParams.update({"figure.dpi": 200})  # increase fig size in notebook
    # # increase figure size
    # fig.set_size_inches(15, 10)


    fig, axs = plt.subplots(nrows=1, ncols=n_cols, sharey=True, sharex=True)
    fig.set_constrained_layout_pads(
        wspace=0.02, # horizontal space between subplots
        hspace=0.01, # vertical space between subplots
        w_pad=0.0, # horizontal padding between subplots and figure edges
        h_pad=0.0, # vertical padding between subplots and figure edges
    )

    target_type = "stock"
    conversation_type = "multi-turn-holiday-v1-default"

    for j, metric_name in enumerate(["sr_target2", "sr_context_strict"]):
        for attack_dir in results[target_type][conversation_type]:
            ax = axs[j]
            if x_data_name == "n_tokens":
                x = results[target_type][conversation_type][attack_dir]["n_tokens"]
            elif x_data_name == "n_turns":
                x = results[target_type][conversation_type][attack_dir]["n_turns"]
            else:
                raise ValueError(f"Invalid x_data_name: {x_data_name}")
            ydata = results[target_type][conversation_type][attack_dir][metric_name]
            if False:  # longer label
                label = ATTACK_SHORT_NAMES[attack_dir.replace("logs/", "")]
            else:
                label_dict = get_short_label(attack_dir.replace("logs/", ""), ablation=ablation)
                label = label_dict["label"]
                color = label_dict["color"]
                marker = label_dict["marker"]
                label = label.replace("Qwen3-VL (", "").replace(")", "")

                # sort by x data
                sorting_mask = np.argsort(x)
                x = np.array(x)[sorting_mask]
                ydata = np.array(ydata)[sorting_mask]
                # now plot
                ax.plot(x, ydata, label=label, color=color, marker=marker)
                if logx:
                    ax.set_xscale("log")
            # formatting
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(PercentFormatter(100))
            ax.grid(visible=True, which="major", axis="both")
            if j == 0: # first col
                pass
            #     target_string = TARGET_NAME_TO_STRING[target_type]
            #     y_label = r"\parbox{3cm}{\textbf{Target:} \textit{" + target_string + r"}"
            #     ax.set_ylabel(y_label, fontsize=6)
                # ax.set_ylabel("Combined success rate")
            else: # not first col
                ax.tick_params(axis='y', which='both', length=0)
            if include_title:
                ax.set_title(f"{conversation_type}-{metric_name}", fontsize=5, pad=1)

    # trim legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, 
        loc="lower left",
        ncol=1, 
        fontsize=6,
        # bbox_to_anchor=(0.43,1.03), # top of first plot
        # bbox_to_anchor=(0.77, 0.5),  # bottom right plot
        bbox_to_anchor=(0.085, 0.15), # bottom left of first plot
        columnspacing=0.25, labelspacing=0.1,
        handletextpad=0.3,
        handlelength=0.9,
        borderpad=0.2,
    )
            
    # plt.tight_layout()
    if ablation == "design":
        fname = f"sr_target2_sr_context_strict_algorithm-ablation_comparison_{x_data_name}"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    plt.savefig(f"output/{fname}.pdf")
    # plt.savefig(f"output/{fname}.png", dpi=200)
    print(f"Saved figure to output/{fname}.pdf")
    # also save to out.pdf
    plt.savefig(f"output/out.pdf")
    plt.close()

def main_plot_token_vs_score_bins():
    target_types = ["stock", "party", "car", "motorola"]
    conversation_types = ["multi-turn-diverse-v1-default", "multi-turn-diversetest-v1-default", "multi-turn-holiday-v1-default"]


    results = {}
    for target_type in target_types:
        results[target_type] = {}
        for conversation_type in conversation_types:
            attack_dirs = MAIN_PAPER_RUNS
            attack_dirs = [el for el in attack_dirs if f"-{target_type}-" in el]
            attack_dirs = [os.path.join("logs", el) for el in attack_dirs]

            results[target_type][conversation_type] = {}
            for attack_dir in attack_dirs:
                results[target_type][conversation_type][attack_dir] = {
                    "combined_score": [],
                    "n_tokens": [],
                    "n_turns": [],
                }
                convs_dir_names = [el for el in os.listdir(attack_dir) if el.startswith(conversation_type)]
                print(f"Found {len(convs_dir_names)} conversations")
                # print(convs_dir_names)

                for convs_dir_name in convs_dir_names:
                    convs_dir_path = os.path.join(attack_dir, convs_dir_name, "conv_token_counts")
                    conv_file_names = [el for el in os.listdir(convs_dir_path) if el.endswith(".json")]
                    print(f"Found {len(conv_file_names)} conversation files")
                    for conv_file_name in conv_file_names:
                        conv_file_path = os.path.join(convs_dir_path, conv_file_name)
                        with open(conv_file_path, "r") as f:
                            conversation = json.load(f)            
                        combined_score = conversation[-1]["combined_score"]
                        n_tokens = conversation[-1]["n_tokens_total"]
                        n_turns = len([m for m in conversation if m["role"] == "user"]) - 2  # start with 0
                        results[target_type][conversation_type][attack_dir]["combined_score"].append(int(combined_score))
                        results[target_type][conversation_type][attack_dir]["n_tokens"].append(int(n_tokens))   
                        results[target_type][conversation_type][attack_dir]["n_turns"].append(int(n_turns))

    # results = sort_results(results)
    lens = [len(results[target_type][conversation_type][attack_dir]["combined_score"]) for target_type in results for conversation_type in results[target_type] for attack_dir in results[target_type][conversation_type]]
    print(f"Number of attacks: {len(lens)}")
    print(f"Number of samples per attack: {lens}")
    print(json.dumps(results, indent=2))
    plot_token_vs_score_bins(results)


def compute_mean_over_paraphrases(results):
    """
    Compute mean and std-dev over -alt3, -alt4, -alt5 paraphrases for each conversation turn.
    Groups conversation types by base name and averages metrics aligned by n_turns.
    """
    import re
    # Mapping from target_type to the string used in conversation directory names
    target_to_conv_name = {
        "motorola": "phone",
    }
    new_results = {}
    
    for target_type in results:
        new_results[target_type] = {}
        conv_name = target_to_conv_name.get(target_type, target_type)
        
        # Group conversation types by base name (e.g., multi-turn-diverse-v1-stock-alt3 -> multi-turn-diverse-v1-default)
        conversation_groups = {}  # base_name -> list of conv_type names
        for conv_type in results[target_type]:
            match = re.match(r'(.+)-' + conv_name + r'-alt[345]$', conv_type)
            if match:
                base = match.group(1) + '-default'
                if base not in conversation_groups:
                    conversation_groups[base] = []
                conversation_groups[base].append(conv_type)
        
        # Compute mean over paraphrases for each group
        for base_name, conv_types in conversation_groups.items():
            new_results[target_type][base_name] = {}
            attack_dirs = list(results[target_type][conv_types[0]].keys())
            
            for attack_dir in attack_dirs:
                new_results[target_type][base_name][attack_dir] = {
                    'sr_combined': [], 'sr_combined_std': [], 'n_tokens': [], 'n_turns': [],
                }
                
                # Collect data indexed by n_turns for alignment
                data_by_turn = {}
                for conv_type in conv_types:
                    data = results[target_type][conv_type][attack_dir]
                    for i, n_turn in enumerate(data['n_turns']):
                        if n_turn not in data_by_turn:
                            data_by_turn[n_turn] = {'sr_combined': [], 'n_tokens': []}
                        data_by_turn[n_turn]['sr_combined'].append(data['sr_combined'][i])
                        data_by_turn[n_turn]['n_tokens'].append(data['n_tokens'][i])
                
                # Compute mean and std for each turn, sorted descending
                for n_turn in sorted(data_by_turn.keys(), reverse=True):
                    scores = data_by_turn[n_turn]['sr_combined']
                    tokens = data_by_turn[n_turn]['n_tokens']
                    new_results[target_type][base_name][attack_dir]['sr_combined'].append(
                        sum(scores) / len(scores)
                    )
                    new_results[target_type][base_name][attack_dir]['sr_combined_std'].append(
                        np.std(scores)
                    )
                    new_results[target_type][base_name][attack_dir]['n_tokens'].append(
                        sum(tokens) / len(tokens)
                    )
                    new_results[target_type][base_name][attack_dir]['n_turns'].append(n_turn)
    
    return new_results


def main(attack_dirs):
    if not ablation:
        target_types = [
            "stock", 
            "party", 
            "car", 
            "motorola",
        ]
        conversation_types = [
            "multi-turn-diverse-v1-default", 
            "multi-turn-diversetest-v1-default", 
            "multi-turn-holiday-v1-default",
        ]
    else:
        target_types = ["stock"]
        conversation_types = ["multi-turn-diverse-v1-default", "multi-turn-diversetest-v1-default", "multi-turn-holiday-v1-default"]

    if paraphrases:
        conversation_types = ["multi-turn-diverse-v1-", "multi-turn-diversetest-v1-", "multi-turn-holiday-v1-"]
        conversation_types = [el + target_type + "-alt" + str(i) for (el, target_type, i) in itertools.product(conversation_types, target_types, [3, 4, 5])]
        conversation_types = [el.replace("motorola", "phone") for el in conversation_types]


    results = {}
    for target_type in target_types:
        results[target_type] = {}
        attack_dirs_cur = [el for el in attack_dirs if f"-{target_type}" in el]
        attack_dirs_cur = [os.path.join("logs", el) for el in attack_dirs_cur]
        for conversation_type in conversation_types:
            results[target_type][conversation_type] = {}
            for attack_dir in attack_dirs_cur:
                results[target_type][conversation_type][attack_dir] = {
                    "sr_combined": [],
                    "sr_target2": [],
                    "sr_context_strict": [],
                    "n_tokens": [],
                    "n_turns": [],
                }
                convs_dir_names = [el for el in os.listdir(attack_dir) if el.startswith(conversation_type)]
                # print(f"Found {len(convs_dir_names)} conversations")
                # print(convs_dir_names)

                for convs_dir_name in convs_dir_names:
                    convs_dir_path = os.path.join(attack_dir, convs_dir_name, "conv_token_counts")
                    eval_results_dir_path = os.path.join(attack_dir, convs_dir_name, "evaluation_results")
                    if not os.path.exists(eval_results_dir_path):
                        print(f"No evaluation results found for {attack_dir}/{convs_dir_name}")
                        continue
                    eval_file_names = [el for el in os.listdir(eval_results_dir_path) if el.endswith(".json")]
                    assert len(eval_file_names) == 1, f"Expected 1 evaluation file, but got {len(eval_file_names)}"
                    with open(os.path.join(eval_results_dir_path, eval_file_names[0]), "r") as f:
                        eval_results = json.load(f)
                    sr_combined = eval_results["evaluation_results"]["success_rate_combined"]
                    sr_target2 = eval_results["evaluation_results"]["success_rate_target2"]
                    sr_context_strict = eval_results["evaluation_results"]["success_rate_context_strict"]
                    # now get average number of tokens per turn
                    conv_file_names = [el for el in os.listdir(convs_dir_path) if el.endswith(".json")]
                    # print(f"Found {len(conv_file_names)} conversation files")
                    n_tokens_list = []
                    n_turns_list = []
                    for conv_file_name in conv_file_names:
                        conv_file_path = os.path.join(convs_dir_path, conv_file_name)
                        with open(conv_file_path, "r") as f:
                            conversation = json.load(f)            
                        n_tokens = conversation[-1]["n_tokens_total"]
                        n_turns = len([m for m in conversation if m["role"] == "user"]) - 2  # start with 0
                        n_tokens_list.append(n_tokens)
                        n_turns_list.append(n_turns)
                    assert len(set(n_turns_list)) == 1, f"Number of turns is not the same for all conversations"
                    n_turns = n_turns_list[0]
                    n_tokens_avg = sum(n_tokens_list) / len(n_tokens_list)
                    results[target_type][conversation_type][attack_dir]["sr_combined"].append(sr_combined)
                    results[target_type][conversation_type][attack_dir]["sr_target2"].append(sr_target2)
                    results[target_type][conversation_type][attack_dir]["sr_context_strict"].append(sr_context_strict)
                    results[target_type][conversation_type][attack_dir]["n_tokens"].append(n_tokens_avg)
                    results[target_type][conversation_type][attack_dir]["n_turns"].append(n_turns)


    if paraphrases:
        results = compute_mean_over_paraphrases(results)

    # results = sort_results(results)
    lens = [len(results[target_type][conversation_type][attack_dir]["sr_combined"]) 
            for target_type in results 
            for conversation_type in results[target_type] 
            for attack_dir in results[target_type][conversation_type]]
    print(f"Number of attacks: {len(lens)}")
    print(f"Number of conversation turns per attack: {lens}")
    # print(json.dumps(results, indent=2))

    # if ablation:
    #     plot_results_ablation(results, x_data_name=x_data_name, logx=logx, include_title=include_title, ablation=ablation)
    # else:
    if single_column:
        plot_results_single_column(results, x_data_name=x_data_name, logx=logx, include_title=include_title, ablation=ablation)
    else:
        plot_results(results, x_data_name=x_data_name, logx=logx, include_title=include_title, ablation=ablation)



if __name__ == "__main__":
    # include_title=False
    include_title=True
    single_column = False

    ####
    paraphrases = False
    ablation = None

    # x_data_name = "n_turns"
    # logx = False

    x_data_name = "n_tokens"
    logx = True

    attack_dirs = MAIN_PAPER_RUNS
    # attack_dirs = TRANSFER_RUNS

    # ablation = "design"
    # attack_dirs = DESIGN_ABLATION_RUNS
    # single_column = True

    # ablation = "iterations"
    # attack_dirs = ITERATION_ABLATION_RUNS

    ####


    main(attack_dirs)

    


