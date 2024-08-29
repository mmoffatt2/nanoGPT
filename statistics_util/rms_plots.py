import os
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import random
import csv
import plotly.graph_objects as go

def calculate_RMS(x):
    batch_dim = (x.shape)[0]
    block_dim = (x.shape)[1]
    embd_dim = (x.shape)[2]
    RMS_first_values = []
    RMS_last_values = []
    RMS_random_values = []
    RMS_individual_values = []
    # batch_idx = random.randint(0, batch_dim-1)
    # block_idx = random.randint(0, block_dim-1)
    batch_idx = 0
    block_idx = 0
    for i in range(1, embd_dim):
        # first selection type
        x_part = x[..., :i]
        rms = x_part.float().norm(2, dim=-1, keepdim=True) / math.sqrt(i)
        RMS_individual_values.append(rms[batch_idx][block_idx][0].detach().numpy())
        RMS_first_values.append(torch.mean(rms).detach().numpy())

        # last selection type
        x_part = x[..., -i:]
        rms = x_part.float().norm(2, dim=-1, keepdim=True) / math.sqrt(i)
        RMS_last_values.append(torch.mean(rms).detach().numpy())

        # random selection type
        indices = torch.randperm(x.size(-1))[:i]
        x_part = x[..., indices]
        rms = x_part.float().norm(2, dim=-1, keepdim=True) / math.sqrt(i)
        RMS_random_values.append(torch.mean(rms).detach().numpy())

    return RMS_first_values, RMS_last_values, RMS_random_values, RMS_individual_values

def table_RMS(values, value_type, ln_name, layer, krmsnorm_num):
    RMS_string = ""
    if krmsnorm_num:
        RMS_string = f"of kRMSNorm where k = {krmsnorm_num}"
    else:
        RMS_string = f"of RMSNorm"
    if layer != None:
        RMS_string = f"{value_type} k values {RMS_string} for {ln_name} in layer {layer}"
    else:
        RMS_string = f"{value_type} k values {RMS_string} for {ln_name}"

    final_value = values[-1]
    with open("rmsruns.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # csv_writer.writerow(["", "Number of elements needed to get RMS within x% of final RMS value"])
        # csv_writer.writerow(["", "Within 15%", "Within 10%", "Within 5%", "Within 1%"])

        # print("final_value: ", final_value)
        k_15 = 0
        k_10 = 0
        k_5 = 0
        k_1 = 0
        for k, val in enumerate(values):
            if val > final_value*0.85 and val < final_value*1.15:
                # print(f"Within 15%, k = {k+1} and val={val}")
                k_15 = k+1
                break
        for k, val in enumerate(values):
            if val > final_value*0.9 and val < final_value*1.1:
                # print(f"Within 10%, k = {k+1} and val={val}")
                k_10 = k+1
                break
        for k, val in enumerate(values):
            if val > final_value*0.95 and val < final_value*1.05:
                # print(f"Within 5%, k = {k+1} and val={val}")
                k_5 = k+1
                break
        for k, val in enumerate(values):
            if val > final_value*0.99 and val < final_value*1.01:
                # print(f"Within 1%, k = {k+1} and val={val}")
                k_1 = k+1
                break
        csv_writer.writerow([RMS_string, k_15, k_10, k_5, k_1])


def graph_RMS(out_dir, tensor, ln_name, timestamp, num_iters, layer=None, krmsnorm_num=None):
    directory_path = os.path.join(out_dir, 'images')
    os.makedirs(directory_path, exist_ok=True)

    RMS_first_values, RMS_last_values, RMS_random_values, RMS_individual_values = calculate_RMS(tensor)
    table_RMS(RMS_first_values, "first", ln_name, layer, krmsnorm_num)
    table_RMS(RMS_last_values, "last", ln_name, layer, krmsnorm_num)
    table_RMS(RMS_random_values, "random", ln_name, layer, krmsnorm_num)
    table_RMS(RMS_individual_values, "individual", ln_name, layer, krmsnorm_num)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(RMS_first_values))),
        y=RMS_first_values,
        mode='lines',
        name=f'avg RMS for first k elements'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(RMS_last_values))),
        y=RMS_last_values,
        mode='lines',
        name=f'avg RMS for last k elements'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(RMS_random_values))),
        y=RMS_random_values,
        mode='lines',
        name=f'avg RMS for random k elements'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(RMS_individual_values))),
        y=RMS_individual_values,
        mode='lines',
        name=f'RMS for first k elements of randomly sampled token'
    ))

    # add titles and legend to Plotly
    if layer == None:
        layer_str = ""
    else:
        layer_str = f"in Layer {layer} "

    fig.update_layout(
        title=f'RMSNorm of {ln_name} {layer_str}after {num_iters} Iterations of Training',
        xaxis_title='k number of elements',
        yaxis_title=f'RMS value',
        height=890,
        width=1200
    )

    if layer == None:
        layer_str = ""
    else:
        layer_str = f"{layer}_"
    fig.write_image(f'{directory_path}/{layer_str}{ln_name}_{num_iters}_RMS_plotly_{timestamp}.png')

def graph_gain(out_dir, gain_stats, max_iter, num_layers, timestamp):
    directory_path = os.path.join(out_dir, 'images')
    os.makedirs(directory_path, exist_ok=True)
    fig = go.Figure()
    # ln_f norm:
    fig.add_trace(go.Scatter(
        x=list(range(max_iter)),
        y=gain_stats["mean_gain_ln_f"],
        mode='lines',
        name='mean gain for ln_f'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(max_iter)),
        y=gain_stats["max_gain_ln_f"],
        mode='lines',
        name='max gain for ln_f'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(max_iter)),
        y=gain_stats["min_gain_ln_f"],
        mode='lines',
        name='min gain for ln_f'
    ))
    fig.update_layout(
        title=f'Change in RMSNorm Gain Values During Training For ln_f',
        xaxis_title='Training Iteration',
        yaxis_title=f'RMSnorm Gain',
        height=890,
        width=1200
    )
    fig.write_image(f'{directory_path}/ln_f_RMSNorm_gain_plotly_{timestamp}.png')
    
    for layer_idx in range(num_layers):
        for name in ['ln_1', 'ln_2']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(max_iter)),
                y=gain_stats[f"mean_gain_{name}"][layer_idx],
                mode='lines',
                name=f'mean gain for Layer {layer_idx} {name}'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(max_iter)),
                y=gain_stats[f"max_gain_{name}"][layer_idx],
                mode='lines',
                name=f'max gain for Layer {layer_idx} {name}'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(max_iter)),
                y=gain_stats[f"min_gain_{name}"][layer_idx],
                mode='lines',
                name=f'min gain for Layer {layer_idx} {name}'
            ))
            fig.update_layout(
                title=f'Change in RMSNorm Gain Values During Training For Layer {layer_idx} {name}',
                xaxis_title='Training Iteration',
                yaxis_title=f'RMSnorm Gain',
                height=890,
                width=1200
            )
            fig.write_image(f'{directory_path}/{layer_idx}_{name}_RMSNorm_gain_plotly_{timestamp}.png')
