from pathlib import Path

neuron_pic_small = str(Path.joinpath(Path(__file__).parent, "neuron_small.png"))
neuron_pic_big = str(Path.joinpath(Path(__file__).parent, "neuron_big.png"))
dark_neuron = str(Path.joinpath(Path(__file__).parent, "darkneuron.png"))
grey_neuron = str(Path.joinpath(Path(__file__).parent, "greyneuron.png"))

if __name__=='__main__':
    from displayarray import breakpoint_display

    breakpoint_display(neuron_pic_small)
    breakpoint_display(neuron_pic_big)