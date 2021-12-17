import torch
import torch.nn.grad
from torch.nn.common_types import _size_2_t
from torch import Tensor
from typing import Optional, List

import tables
from typing import Union

class Layer(tables.IsDescription):
    name = tables.StringCol(16)
    idnumber = tables.Int64Col()
    input_shape = tables.UInt64Col()
    output_shape = tables.UInt64Col()


class Neuron(tables.IsDescription):
    idnumber = tables.Int64Col()
    layer_id = tables.Int64Col()
    activation = tables.Float32Col()


class Synapse(tables.IsDescription):
    input_id = tables.Int64Col()
    neuron_id = tables.Int64Col()
    multiply_input = tables.Float32Col()  # if connected, multiply value; if not, connection percent.
    is_excitatory = tables.BoolCol()  # True = excitatory, False = inhibitory
    is_connected = tables.BoolCol()  # True = connected, False = searching


class SparseLinear(torch.nn.Module):
    def __init__(
            self,
            input_shape,
            output_shape,
            name: str,
            file: Union[str, tables.File] = None,
            device=None,
            dtype=None
    ):
        super(SparseLinear, self).__init__()

        self.name = name

        if isinstance(file, str):
            self.file = tables.open_file(self.filename, mode="a", title=self._gen_title())
            self.filename = file
        else:
            self.file = file
            self.filename = None

        self.group = self.file.create_group("/", name)
        #if self.group is None:
        #    self.group = self.file.create_group("/", name)

        self.neurons = self.file.create_table(self.group, 'neurons')
        self.synapses = self.file.create_table(self.group, 'synapses')
        #if 'neurons' not in self.file.create_table()
        #table = h5file.create_table(group, 'readout', Particle, "Readout example")

        self.restrict_parameters()

    def reset(self):
        neurons:tables.Table.row = self.neurons.where(f"""activation!=0""")
        for n in neurons:
            n['activation'] = 0
            n.update()
        self.neurons.flush()

    def forward(self, inp_dict, threshold=0.5, reset=True):
        if reset:
            self.reset()

        synapses = set()
        for ii in inp_dict.indices():
            synapses.add(self.synapses.where(f"""(input_id=={ii}) & (is_connected)"""))

        neurons = dict()
        updates = dict()
        for s in synapses:
            update_val = (1 if s['is_excitatory'] else -1) * s['multiply_input'] * inp_dict[s['input_id']]
            if s['neuron_id'] not in updates:
                updates[s['neuron_id']] = update_val
            else:
                updates[s['neuron_id']] += update_val
            if s['neuron_id'] not in neurons:
                neurons[s['neuron_id']] = self.neurons.where(f"""idnumber=={s['neuron_id']}""")

        for i, n in neurons.items():
            neurons[i]['activation'] += updates[i]
            neurons[i].update()

        self.neurons.flush()

        out = self.neurons.where(f"""activation>{threshold}""")
        out_dict = dict()
        for o in out:
            out_dict[o['idnumber']] = o['activation']

        return out_dict

    def backward(self, inp_dict):







    def full_output(self):
        pass

    def select_output(self):
        pass


    def _gen_title(self):
        return type(self).__name__

    def restrict_parameters(self):
        eps = 1e-6
        disteps = 1e-2

        with torch.no_grad():
            # mimic physical weights. You can't be more than 100% connected, or nan% connected
            max = torch.max(torch.abs(self.weight)) + eps
            if max > 1.0 + disteps:
                self.weight = torch.nn.Parameter(self.weight / max)

            if self.bias is not None:
                max = torch.max(torch.abs(self.bias)) + eps
                if max > 1.0 + disteps:
                    self.bias = torch.nn.Parameter(self.bias / max)

    def forward(self, input: Tensor) -> Tensor:
        eps = 1e-6
        disteps = 1e-2

        if self.training:
            self.restrict_parameters()

        f = super(NormLinear, self).forward(input)
        f = f / (torch.max(torch.abs(f)) + eps)

        return f
