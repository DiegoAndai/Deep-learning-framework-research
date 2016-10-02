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
    groundtruth = testset.label[i]  -- actual digit
    prediction = net:forward(testset.data[i])  -- predicted digit (log probabilities)
    confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print('Hits: ' .. correct, 'Accuracy: ' .. 100*correct/10000 .. ' % ') -- accuracy


-- precision and recall

avg_recall, avg_precision = 0, 0
for cls=1,#classes do
	tp, fn, fp = 0, 0 ,0
	-- code repetition (omg!)
	for i=1,10000 do
		groundtruth = testset.label[i]  -- actual digit
		prediction = net:forward(testset.data[i])  -- predicted digit (log probabilities)
		confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order


		if groundtruth == cls then  -- if the actual digit is of the class cls
		    
		    if groundtruth == indices[1] then  -- if prediction was correct
		        tp = tp + 1
		    else  -- if prediction was incorrect
		    	fn = fn + 1
		    end

		else  -- if the actual digit is not of class cls

			if indices[1] == cls then -- and class cls was predicted
				fp = fp + 1
			end

		end
	end
	precision = tp/(tp + fp)
	recall = tp/(tp + fn)
	avg_precision = avg_precision + precision
	avg_recall = avg_recall + recall

	print("Class " .. cls - 1 .. ":\nPrecision: " .. precision .. "\nRecall: " .. recall)
end

avg_recall = avg_recall/#classes
avg_precision = avg_precision/#classes
print("Average precision: " .. avg_precision)
print("Average recall: " .. avg_recall)

