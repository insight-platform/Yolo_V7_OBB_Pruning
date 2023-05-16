import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
import copy
from models.yolo import Model
from torch_pruning import MetaPruner
from utils.torch_utils import intersect_dicts, is_parallel
from yolov7obb.models.common import RepConv
from yolov7obb.models.yolo import IDetect


def load_model(weights, device, ref_model: bool = False):
    if ref_model:
        model = torch.load(weights, map_location=device)
    else:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(ckpt['model'].yaml).to(device)  # create
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
        assert len(state_dict) == len(model.state_dict())

    model.float()
    model.model[-1].export = True
    return model


def progressive_pruning(
        pruner: MetaPruner,
        model: nn.Module,
        model_ref: nn.Module,
        operation_reduce: float,
        example_inputs: torch.Tensor
):
    model.eval()
    model_ref.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_param_reduce = 1
    iter = 0
    while current_param_reduce < operation_reduce:
        iter += 1
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_param_reduce = float(base_ops) / pruned_ops
        print(f"iter {iter}: base operations = {base_ops}, operation after pruning={pruned_ops}, decreased operations in={current_param_reduce}")
    return current_param_reduce


def channel_prune(
        trained_model: nn.Module,
        initial_model: nn.Module,
        example_inputs: torch.Tensor,
        operation_reduce: float = 1.2,
        round_to: int = 32
):
    model = copy.deepcopy(trained_model)
    model.cpu().eval()

    ignored_layers = []

    exclude = (RepConv, IDetect)
    exclude_idx = [102, 103, 104, 105]
    for i, layer in enumerate(model.model):
        if isinstance(layer, exclude) or i in exclude_idx:
            ignored_layers.append(layer)

    initila_size = tp.utils.count_params(model)

    imp = tp.importance.BNScaleImportance()

    ch_sparsity_dict = {}
    unwrapped_parameters = []

    model.eval()
    initial_model.eval()
    pruner = tp.pruner.BNScalePruner(
        model=model,
        example_inputs=example_inputs,
        importance=imp,
        iterative_steps=400,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=1,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        global_pruning=False,
        model_ref=initial_model,
        round_to=round_to
    )

    progressive_pruning(pruner, model, initial_model, operation_reduce=operation_reduce, example_inputs=example_inputs)

    model.train()
    initial_model.train()
    trained_model.train()
    with torch.no_grad():
        out_prune = model(example_inputs)
        out_ori = trained_model(example_inputs)
        pruned_size = tp.utils.count_params(model)
        print(f"Params: {initila_size} => {pruned_size}, param_reduce={initila_size/pruned_size}")
        if isinstance(out_prune, (list, tuple)):
            for o, o2 in zip(out_prune, out_ori):
                print("  Output: ", o.shape)
                assert o.shape == o2.shape, f'{o.shape} {o2.shape}'
        else:
            print("  Output: ", out_prune.shape)
    return model, initial_model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='The path to the weights of the trained model')
    parser.add_argument('--weights-ref', type=str, help='Path to weights at which the model was trained')
    parser.add_argument('-o', '--operation-reduce', default=1.1, type=float,
                        help='Sets by how many times the number of parameters in the model is reduced.')
    parser.add_argument('-r', '--round-to', default=32, type=int,
                        help='Channel rounding when pruning')
    parser.add_argument('--shape', nargs='+', type=int, default=[1, 3, 640, 640],
                        help='The size of the input vector of the model')

    opt = parser.parse_args()
    trained_weights = Path(opt.weights)
    initial_weights = Path(opt.weights_ref)
    device = torch.device('cpu')
    trained_model = load_model(trained_weights, device)
    initial_model = load_model(initial_weights, device, ref_model=True)
    example_inputs = torch.zeros(opt.shape, dtype=torch.float32).to(device)
    trained_pruned_model, initial_pruned_model = channel_prune(
        trained_model=trained_model,
        initial_model=initial_model,
        example_inputs=example_inputs,
        operation_reduce=opt.operation_reduce,
        round_to=opt.round_to
    )
    trained_pruned_model.model[-1].export = False
    initial_pruned_model.model[-1].export = False
    ckpt = {
        'model': copy.deepcopy(trained_pruned_model.module if is_parallel(trained_pruned_model) else trained_pruned_model).half(),
        'optimizer': None,
        'epoch': -1,
    }
    ckpt_ref = {
        'model': copy.deepcopy(initial_pruned_model.module if is_parallel(initial_pruned_model) else initial_pruned_model).half(),
        'optimizer': None,
        'epoch': -1,
    }
    torch.save(ckpt, trained_weights.parent / f"{trained_weights.stem}_prune_{opt.operation_reduce}_{opt.round_to}.pt")
    print("Saved", trained_weights.parent / f"{trained_weights.stem}_prune_{opt.operation_reduce}_{opt.round_to}.pt")
    torch.save(ckpt_ref, trained_weights.parent / f"{trained_weights.stem}_prune_{opt.operation_reduce}_{opt.round_to}_init.pt")
    print("Saved", trained_weights.parent / f"{trained_weights.stem}_prune_{opt.operation_reduce}_{opt.round_to}_init.pt")


if __name__ == '__main__':
    main()
