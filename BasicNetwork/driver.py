from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(2, 3, 1)
ds = SupervisedDataSet(2, 1)

# Study hours, sleep hours
ds.addSample((3, 5), (75,))
ds.addSample((5, 1), (62,))
ds.addSample((10, 5), (86,))
ds.addSample((8, 9), (92,))

trainer = BackpropTrainer(net, ds)

print "Training..."
trainer.trainUntilConvergence()

print net.activate([3, 5])
