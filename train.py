from network.network_v1 import Network

net = Network()
print("Training...")
net.train(steps=100)
print("Exporting...")
net.export()