require 'paths'

parser = {}

dataset = {}
function dataset:new(dt)
	function dt:size()
		return self.data:size(1) -- return the length of the data tensor. 
	end

	setmetatable(dt, self)
	self.__index = function(t, i) 
						return {t.data[i], t.label[i]} 
                	end
	return dt
end


--[[The following code is a modified version of the code at
https://github.com/andresy/mnist/blob/master/init.lua"""]]


function readlush(filename)

	local f = torch.DiskFile(filename)
	f:bigEndianEncoding()
	f:binary()
	local ndim = f:readInt() - 0x800
	assert(ndim > 0)
	local dims = torch.LongTensor(ndim)
	for i=1,ndim do
	  dims[i] = f:readInt()
	  assert(dims[i] > 0)
	end
	local nelem = dims:prod(1):squeeze()
	local data = torch.ByteTensor(dims:storage())
	f:readByte(data:storage())
	f:close()
	return data
end

function createdataset(dataname, labelname, one_hot, n_classes)

	local data = readlush(dataname):double()
	local label = readlush(labelname):double():add(1)
	assert(data:size(1) == label:size(1))

	if one_hot then
		indices = label:view(-1,1):long()
		one_hot = torch.zeros(label:size(1), n_classes)
		label = one_hot:scatter(2, indices:long(), 1)
	end

	local dataset = {data=data, label=label}

	return dataset
end

function parser.traindataset(one_hot, n_classes)
	local path = paths.dirname( paths.thisfile() )
	return dataset:new(createdataset(paths.concat(path, 'data/train-images-idx3-ubyte'),
		paths.concat(path, 'data/train-labels-idx1-ubyte'), one_hot, n_classes))
end

function parser.testdataset(one_hot, n_classes)
	local path = paths.dirname( paths.thisfile() )
	
	return dataset:new(createdataset(paths.concat(path, 'data/t10k-images-idx3-ubyte'),
		paths.concat(path, 'data/t10k-labels-idx1-ubyte'), one_hot, n_classes))
end

return parser
