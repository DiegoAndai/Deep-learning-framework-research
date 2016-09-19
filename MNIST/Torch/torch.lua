require 'nn'

-- Usage of mnist loader 
--local mnist = require 'mnist'
--local trainset = mnist.traindataset()
--local testset = mnist.testdataset()
--classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


require 'paths'
if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}


setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end



net = nn.Sequential()  -- sequential nn.

net:add(nn.Reshape(3*32*32))  -- flatten images. (could use View)
net:add(nn.Linear(3*32*32, #classes))  -- Fully connected layer.
net:add(nn.LogSoftMax())  -- Softmax layer.


criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.trainer:train(trainset)trainer:train(trainset)

trainer:train(trainset)

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor


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


