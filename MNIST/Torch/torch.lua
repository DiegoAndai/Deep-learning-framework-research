require 'nn'

local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

classes = {'1','2','3','4','5','6','7','8','9','10'}

net = nn.Sequential()  -- sequential nn.

net:add(nn.Reshape(28*28))  -- flatten images.
net:add(nn.Linear(28*28, #classes))  -- Fully connected layer.
net:add(nn.LogSoftMax())  -- Softmax layer.
criterion = nn.CrossEntropyCriterion()
