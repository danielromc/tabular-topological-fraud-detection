import torch
import numpy as np
import json
import sys

path = 'artifacts_graph/graph_data.pt'
try:
    data = torch.load(path)
except Exception as e1:
    try:
        data = torch.load(path, weights_only=False)
    except Exception as e2:
        print('ERROR_LOADING:', e1)
        print('ERROR_LOADING2:', e2)
        sys.exit(2)

print('TOP_LEVEL_KEYS:', list(data.keys()))
print('HeteroData node_types:', getattr(data, 'node_types', None))
print('HeteroData edge_types:', getattr(data, 'edge_types', None))
for k in list(data.keys()):
    v = data[k]
    try:
        t = type(v)
    except Exception:
        t = str(v)
    info = ''
    if isinstance(v, torch.Tensor):
        info = f'shape={tuple(v.size())}'
    else:
        if hasattr(v, 'size'):
            try:
                info = f'size={v.size()}'
            except Exception:
                pass
        if hasattr(v, 'keys'):
            try:
                info = info + f' keys={list(v.keys())}'
            except Exception:
                pass
        if hasattr(v, 'to_dict'):
            try:
                info = info + f' to_dict_keys={list(v.to_dict().keys())}'
            except Exception:
                pass
    print(f"- {k}: {t} {info}")
# Additional diagnostics for data['x'] NodeStorage
if 'x' in data:
    nx = data['x']
    try:
        print('data["x"].num_nodes:', getattr(nx, 'num_nodes', None))
    except Exception:
        pass
    try:
        print('data["x"].num_node_features:', getattr(nx, 'num_node_features', None))
    except Exception:
        pass
    try:
        print('data["x"].node_attrs:', getattr(nx, 'node_attrs', None))
    except Exception:
        pass

# Try common locations for claim node features
node_types = getattr(data, 'node_types', []) or []
if 'claim' in node_types:
    x_claim = data['claim'].x
elif 'x' in data and isinstance(data['x'], torch.Tensor):
    x_claim = data['x']
elif 'x' in data and hasattr(data['x'], 'to_dict'):
    # attempt to find a tensor inside NodeStorage
    nd = data['x']
    for key in nd.keys():
        cand = nd[key]
        if isinstance(cand, torch.Tensor):
            x_claim = cand
            break
    else:
        print('Could not find tensor inside data["x"] NodeStorage; aborting.')
        sys.exit(2)
else:
    print('Could not locate claim features. Aborting.')
    sys.exit(2)

# Additional diagnostics for data['x'] NodeStorage
if 'x' in data:
    nx = data['x']
    try:
        print('data["x"].num_nodes:', getattr(nx, 'num_nodes', None))
    except Exception:
        pass
    try:
        print('data["x"].num_node_features:', getattr(nx, 'num_node_features', None))
    except Exception:
        pass
    try:
        print('data["x"].node_attrs:', getattr(nx, 'node_attrs', None))
    except Exception:
        pass
if hasattr(x_claim, 'cpu'):
    x_claim = x_claim.cpu()
    if not isinstance(x_claim, torch.Tensor):
        try:
            keys = list(x_claim.keys())
            print('DEBUG_NODESTORAGE_KEYS:', keys)
            for k in keys:
                v = x_claim[k]
                if isinstance(v, torch.Tensor):
                    x_claim = v
                    break
                if hasattr(v, 'x'):
                    x_claim = v.x
                    break
        except Exception as e:
            print('ERROR_INSPECTING_NODESTORAGE:', e)
            # extra diagnostics
            print('TYPE:', type(x_claim))
            try:
                print('REPR:', repr(x_claim))
            except Exception:
                pass
            try:
                print('DIR:', dir(x_claim)[:200])
            except Exception:
                pass
            if hasattr(x_claim, 'to_dict'):
                try:
                    d = x_claim.to_dict()
                    print('TO_DICT_KEYS:', list(d.keys()))
                except Exception:
                    pass
            sys.exit(2)

    if isinstance(x_claim, torch.Tensor):
        if hasattr(x_claim, 'cpu'):
            x_claim = x_claim.cpu()
        x_claim = x_claim.numpy()
    else:
        print('ERROR: after extraction x_claim is not a torch.Tensor. TYPE:', type(x_claim))
        try:
            print('REPR:', repr(x_claim))
        except Exception:
            pass
        try:
            print('DIR:', dir(x_claim)[:200])
        except Exception:
            pass
        sys.exit(2)

print('Rango por feature (top 10 más extremas):')
ranges = x_claim.max(axis=0) - x_claim.min(axis=0)
order = np.argsort(ranges)[::-1]

with open('artifacts/feature_names.json') as f:
    fnames = json.load(f)

for i in order[:10]:
    print(f"  {fnames[i]:30s}  min={x_claim[:,i].min():.2f}  max={x_claim[:,i].max():.2f}  std={x_claim[:,i].std():.2f}")
