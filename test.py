#coding = utf-8
import mnist_loader
import network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net=network.Network([784.100,10])
net.SGD(training_data, 30, 10, 2.5, test_data=test_data)
