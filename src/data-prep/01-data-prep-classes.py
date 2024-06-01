import json 


d = {}
with open("datasets/raw/d2-kaggle/classes.json") as classes_file:
    d = json.loads(classes_file.read())
print(d)

l  = list(d.keys())
print(l)


final_l = ['junction', 'crossover', 'terminal', 'gnd', 'vss', 'voltage.dc', 'voltage.ac', 'voltage.battery', 'resistor', 'resistor.adjustable', 'resistor.photo', 'capacitor.unpolarized', 'capacitor.polarized', 'capacitor.adjustable', 'inductor', 'inductor.ferrite', 'inductor.coupled', 'transformer', 'diode', 'diode.light_emitting', 'diode.thyrector', 'diode.zener', 'diac', 'triac', 'thyristor', 'varistor', 'transistor.bjt', 'transistor.fet', 'transistor.photo', 'operational_amplifier', 'operational_amplifier.schmitt_trigger', 'optocoupler', 'integrated_circuit', 'integrated_circuit.ne555', 'integrated_circuit.voltage_regulator', 'xor', 'and', 'or', 'not', 'nand', 'nor', 'probe', 'probe.current', 'probe.voltage', 'switch', 'relay', 'socket', 'fuse', 'speaker', 'motor', 'lamp', 'microphone', 'antenna', 'crystal', 'mechanical', 'magnetic', 'optical', 'block']

