require 'nn'
require 'paths'

parser = dofile 'MNISTParser.lua'

local trainset = parser.traindataset()
local testset = parser.testdataset()
classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}


-- nn definition

net = nn.Sequential()  -- sequential nn.

net:add(nn.Reshape(28*28))  -- flatten images. (could use View)
net:add(nn.Linear(28*28, #classes))  -- Fully connected layer.
net:add(nn.LogSoftMax())  -- Softmax layer.


--training
criterion = nn.CrossEntropyCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

trainer:train(trainset)


--testing
correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ') -- accuracy
--]]