"""
This module contains wrappers for PyTorch models to output both the last layer and a specified intermediate layer.
This is used for SplitConfidence which can redirect the different outputs to different confidence estimation modules.
"""
import weakref

import torch
import torch.nn as nn


class ModelInputOutputWrapper(nn.Module):
    """
    Wrapper for a PyTorch model to output both the last layer and multiple intermediate layers input, output, or both.
    """

    def __init__(self,
                 model,
                 target_layer_names,
                 flatten=False,
                 concat=False,
                 capture_modes='output',
                 entry_indices=0,
                 return_final: bool = True,
                 return_y: bool = False,
                 feature_reducers: dict = None
                 ):
        """
        Wrapper for a PyTorch model to output both the last layer and specified intermediate layers' inputs, outputs, or both.

        Args:
            model (nn.Module)
            target_layer_names (str|list)
            flatten (bool)
            concat (bool)
            capture_modes (str|list) 'input'|'output'|'both'
            entry_indices (int|list)
            return_final (bool): include model final output in returned tuple (default True)
            return_y (bool): include provided y (passed to forward) in returned tuple (default False)
            feature_reducers (dict): optional per-(layer,mode) reducers applied after flattening
        """
        super(ModelInputOutputWrapper, self).__init__()
        self.model = model
        self.flatten = flatten
        self.concat = concat
        self.output_tuple = True
        self.return_final = return_final
        self.return_y = return_y
        self.feature_reducers = feature_reducers or {}

        if isinstance(target_layer_names, str):
            self.target_layer_names = [target_layer_names]
            self.output_tuple = False
        else:
            self.target_layer_names = target_layer_names

        if isinstance(capture_modes, str):
            self.capture_modes = [capture_modes] * len(self.target_layer_names)
        else:
            if len(capture_modes) != len(self.target_layer_names):
                raise ValueError("capture_modes must have the same length as target_layer_names")
            self.capture_modes = capture_modes

        if isinstance(entry_indices, int):
            self.entry_indices = [entry_indices] * len(self.target_layer_names)
        else:
            if len(entry_indices) != len(self.target_layer_names):
                raise ValueError("entry_indices must have the same length as target_layer_names")
            self.entry_indices = entry_indices

        self.layer_data = list(zip(self.target_layer_names, self.capture_modes, self.entry_indices))
        self.target_layer_outputs = {}
        self.target_layer_inputs = {}
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Use weak ref to break cycles: module -> hook fn -> self -> model -> module. Allows GC.
        weak_self = weakref.ref(self)

        def create_hook_input(weak_self, layername, entry_idx):
            def hook_fn(_, inputs, outputs):
                s = weak_self()
                if s is None:
                    return
                s.target_layer_inputs[layername] = inputs[entry_idx]

            return hook_fn

        def create_hook_output(weak_self, layername):
            def hook_fn(_, inputs, outputs):
                s = weak_self()
                if s is None:
                    return
                s.target_layer_outputs[layername] = outputs

            return hook_fn

        found_layers = set()
        for layer_name, mode, entry_idx in self.layer_data:
            # Find target module
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name:
                    module = mod
                    break

            if not module:
                raise ValueError(f"Layer '{layer_name}' not found in model")

            if mode == 'input' or mode == 'both':
                h = module.register_forward_hook(create_hook_input(weak_self, layer_name, entry_idx))
                self._hook_handles.append(h)
            if mode == 'output' or mode == 'both':
                h = module.register_forward_hook(create_hook_output(weak_self, layer_name))
                self._hook_handles.append(h)
            found_layers.add(layer_name)

        if len(set(self.target_layer_names)) != len(set(found_layers)):
            missing = set(self.target_layer_names) - found_layers
            raise ValueError(f"Layers {missing} not found in model")

    def forward(self, x, y=None):
        """
        Forward pass.
        For each (layer, mode):
          mode 'input'  -> append input tensor
          mode 'output' -> append output tensor
          mode 'both'   -> append input THEN output
        Ordering matches the order of target_layer_names / capture_modes with 'both' expanded.
        """
        try:
            self.target_layer_inputs.clear()
            self.target_layer_outputs.clear()

            final_output = self.model(x)

            outputs_list = []
            for layer_name, mode, _entry_idx in self.layer_data:
                if mode in ('input', 'both'):
                    tensor_in = self.target_layer_inputs[layer_name]
                    if isinstance(tensor_in, tuple):
                        tensor_in = tensor_in[0]
                    key = (layer_name, 'input')
                    if key in self.feature_reducers:
                        red = self.feature_reducers[key]
                        tensor_in = red(tensor_in.float())
                    elif self.flatten:
                        b = tensor_in.shape[0]
                        tensor_in = tensor_in.reshape(b, -1)
                    outputs_list.append(tensor_in)

                if mode in ('output', 'both'):
                    tensor_out = self.target_layer_outputs[layer_name]
                    if isinstance(tensor_out, tuple):
                        tensor_out = tensor_out[0]
                    key = (layer_name, 'output')
                    if key in self.feature_reducers:
                        red = self.feature_reducers[key]
                        tensor_out = red(tensor_out.float())
                    elif self.flatten:
                        b = tensor_out.shape[0]
                        tensor_out = tensor_out.reshape(b, -1)
                    outputs_list.append(tensor_out)

            if self.flatten:
                for i, t in enumerate(outputs_list):
                    b = t.shape[0]
                    outputs_list[i] = t.reshape(b, -1)

            if self.concat:
                embeddings = torch.cat(outputs_list, dim=-1)
            else:
                if not self.output_tuple and len(outputs_list) == 1:
                    embeddings = outputs_list[0]
                else:
                    embeddings = outputs_list

            if self.return_final and self.return_y:
                ret = (embeddings, final_output, y)
            elif self.return_final:
                ret = (embeddings, final_output)
            elif self.return_y:
                ret = (embeddings, y)
            else:
                ret = embeddings
        except Exception as e:
            raise e
        finally:
            self.target_layer_inputs.clear()
            self.target_layer_outputs.clear()

        return ret

    def clear(self):
        """Explicitly clear hooks and feature reducers to break reference cycles."""
        self.remove_hooks()
        if self.feature_reducers:
            for key in list(self.feature_reducers.keys()):
                del self.feature_reducers[key]
            self.feature_reducers.clear()
        self.feature_reducers = None

    def remove_hooks(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def __del__(self):
        self.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove_hooks()
        return False


class ModelInputOutputWrapperOnDemand(nn.Module):
    """
    On-demand wrapper for a PyTorch model to output both the last layer and multiple intermediate layers input, output, or both.
    Hooks are registered only during forward and removed immediately after.
    """

    def __init__(self,
                 model,
                 target_layer_names,
                 flatten=False,
                 concat=False,
                 capture_modes='output',
                 entry_indices=0,
                 return_final: bool = True,
                 return_y: bool = False,
                 feature_reducers: dict = None
                 ):
        """
        Same as ModelInputOutputWrapper but hooks are not persistent.
        """
        super(ModelInputOutputWrapperOnDemand, self).__init__()
        self.model = model
        self.flatten = flatten
        self.concat = concat
        self.output_tuple = True
        self.return_final = return_final
        self.return_y = return_y
        self.feature_reducers = feature_reducers or {}

        if isinstance(target_layer_names, str):
            self.target_layer_names = [target_layer_names]
            self.output_tuple = False
        else:
            self.target_layer_names = target_layer_names

        if isinstance(capture_modes, str):
            self.capture_modes = [capture_modes] * len(self.target_layer_names)
        else:
            if len(capture_modes) != len(self.target_layer_names):
                raise ValueError("capture_modes must have the same length as target_layer_names")
            self.capture_modes = capture_modes

        if isinstance(entry_indices, int):
            self.entry_indices = [entry_indices] * len(self.target_layer_names)
        else:
            if len(entry_indices) != len(self.target_layer_names):
                raise ValueError("entry_indices must have the same length as target_layer_names")
            self.entry_indices = entry_indices

        self.layer_data = list(zip(self.target_layer_names, self.capture_modes, self.entry_indices))

    def forward(self, x, y=None):
        # Register hooks on-demand
        target_layer_outputs = {}
        target_layer_inputs = {}
        hook_handles = []

        weak_self = weakref.ref(self)

        def create_hook_input(weak_self, layername, entry_idx):
            def hook_fn(_, inputs, outputs):
                s = weak_self()
                if s is None:
                    return
                target_layer_inputs[layername] = inputs[entry_idx]

            return hook_fn

        def create_hook_output(weak_self, layername):
            def hook_fn(_, inputs, outputs):
                s = weak_self()
                if s is None:
                    return
                target_layer_outputs[layername] = outputs

            return hook_fn

        found_layers = set()
        for layer_name, mode, entry_idx in self.layer_data:
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name:
                    module = mod
                    break
            if not module:
                raise ValueError(f"Layer '{layer_name}' not found in model")

            if mode == 'input' or mode == 'both':
                h = module.register_forward_hook(create_hook_input(weak_self, layer_name, entry_idx))
                hook_handles.append(h)
            if mode == 'output' or mode == 'both':
                h = module.register_forward_hook(create_hook_output(weak_self, layer_name))
                hook_handles.append(h)
            found_layers.add(layer_name)

        if len(set(self.target_layer_names)) != len(set(found_layers)):
            missing = set(self.target_layer_names) - found_layers
            raise ValueError(f"Layers {missing} not found in model")

        try:
            # Forward pass
            final_output = self.model(x)

            outputs_list = []
            for layer_name, mode, _entry_idx in self.layer_data:
                if mode in ('input', 'both'):
                    tensor_in = target_layer_inputs[layer_name]
                    if isinstance(tensor_in, tuple):
                        tensor_in = tensor_in[0]
                    key = (layer_name, 'input')
                    if key in self.feature_reducers:
                        red = self.feature_reducers[key]
                        tensor_in = red(tensor_in.float())
                    elif self.flatten:
                        b = tensor_in.shape[0]
                        tensor_in = tensor_in.reshape(b, -1)
                    outputs_list.append(tensor_in)

                if mode in ('output', 'both'):
                    tensor_out = target_layer_outputs[layer_name]
                    if isinstance(tensor_out, tuple):
                        tensor_out = tensor_out[0]
                    key = (layer_name, 'output')
                    if key in self.feature_reducers:
                        red = self.feature_reducers[key]
                        tensor_out = red(tensor_out.float())
                    elif self.flatten:
                        b = tensor_out.shape[0]
                        tensor_out = tensor_out.reshape(b, -1)
                    outputs_list.append(tensor_out)

            if self.flatten:
                for i, t in enumerate(outputs_list):
                    b = t.shape[0]
                    outputs_list[i] = t.reshape(b, -1)

            if self.concat:
                embeddings = torch.cat(outputs_list, dim=-1)
            else:
                if not self.output_tuple and len(outputs_list) == 1:
                    embeddings = outputs_list[0]
                else:
                    embeddings = outputs_list

            if self.return_final and self.return_y:
                ret = (embeddings, final_output, y)
            elif self.return_final:
                ret = (embeddings, final_output)
            elif self.return_y:
                ret = (embeddings, y)
            else:
                ret = embeddings

        finally:
            # Remove hooks immediately
            for h in hook_handles:
                try:
                    h.remove()
                except Exception:
                    pass

        return ret


if __name__ == '__main__':
    import time

    num_runs = 100  # Reduced for larger model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA not available; benchmarking on CPU.")

    # Bigger model: ResNet18
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    )

    # Bigger batch size: 32, with input shape for ResNet (3, 224, 224)
    x = torch.randn(32, 3, 224, 224).to(device)

    # Wrapper 1: Persistent hooks
    wrapper1 = ModelInputOutputWrapper(
        model,
        target_layer_names=['0', '2'],  # conv1 and layer1.0.conv1 in ResNet18
        flatten=False,
        concat=False,
        capture_modes=['output', 'output'],  # Adjusted for output capture
        entry_indices=[0, 0],
    ).to(device)

    # Wrapper 2: On-demand hooks
    wrapper2 = ModelInputOutputWrapperOnDemand(
        model,
        target_layer_names=['0', '2'],
        flatten=False,
        concat=False,
        capture_modes=['output', 'output'],
        entry_indices=[0, 0],
    ).to(device)

    # Benchmark wrapper1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = wrapper1(x)
        end.record()
        torch.cuda.synchronize()
        time1 = start.elapsed_time(end) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = wrapper1(x)
        time1 = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark wrapper2
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = wrapper2(x)
        end.record()
        torch.cuda.synchronize()
        time2 = start.elapsed_time(end) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = wrapper2(x)
        time2 = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark wrapper1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = wrapper1(x)
        end.record()
        torch.cuda.synchronize()
        time1 = start.elapsed_time(end) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = wrapper1(x)
        time1 = (time.time() - start) / num_runs * 1000  # ms

    # Benchmark wrapper2
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = wrapper2(x)
        end.record()
        torch.cuda.synchronize()
        time2 = start.elapsed_time(end) / num_runs
    else:
        start = time.time()
        for _ in range(num_runs):
            _ = wrapper2(x)
        time2 = (time.time() - start) / num_runs * 1000  # ms

    print(f"Benchmark results ({num_runs} runs on {device}):")
    print(f"  ModelInputOutputWrapper (persistent hooks): {time1:.3f} ms per forward")
    print(f"  ModelInputOutputWrapperOnDemand (on-demand hooks): {time2:.3f} ms per forward")
