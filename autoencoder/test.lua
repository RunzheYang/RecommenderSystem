require 'torch'
require 'nn'
require 'optim'
require 'nnsparse'
require 'xlua'
require 'data_loader'

local user_num = 94317
local topic = 2048
local batch = 256
local epoch = 20

local sgdConfiguration = {
	learningRate      = 0.9, 
	-- learningRateDecay = 0.35,
	-- weightDecay       = 0.03,
}

-- load data
local dataset = Dataset:new(nil, "item_recom/train_info.tsv", 0.9)
local train = dataset.train
local valid = dataset.valid 

-- build network
-- local net = nn.Sequential()
-- net:add(nnsparse.SparseLinearBatch(user_num, topic))
-- net:add(nn.Tanh())
-- net:add(nn.Linear(topic, user_num))
-- net:add(nn.Tanh())

local net = torch.load('models/ae_simple_32.models')

print(net)

-- define the loss function
-- local loss_func = nnsparse.SDAESparseCriterion(nn.MSECriterion(), {
-- local loss_func = nnsparse.MaskCriterion(nn.MSECriterion(), {
-- 		alpha = 1,
-- 		beta = 0.5,
-- 		hideRatio = 0.25,
-- 	})
-- loss_func.inputDim = user_num
-- loss_func.sizeAverage = false

-- local function batchify(data)
-- 	local input, minibatch = {}, {}
-- 	local shuffle = torch.randperm(user_num)
-- 	shuffle:apply(function(k) 
-- 			if data[k] then
-- 				input[#input + 1] = data[k]
-- 				if #input == batch then
-- 					minibatch[#minibatch + 1] = input
-- 					input = {}
-- 				end
-- 			end
-- 		end)
-- 	if #input > 0 then
-- 		minibatch[#minibatch + 1] = input
-- 		input = {}
-- 	end
-- 	return minibatch
-- end


-- local function trainAE(net, epc)

-- 	-- create minibatch
-- 	local input, minibatch = {}, batchify(train)

-- 	local w, dw = net:getParameters()
-- 	for cnt, input in pairs(minibatch) do
-- 		xlua.progress(cnt, #minibatch)
-- 		local function feval(x)
-- 			-- reset gradients and losses
-- 			net:zeroGradParameters()
-- 			-- set target as input
-- 			local target = input
-- 			-- add noise to input
-- 			local noisy_input = loss_func:prepareInput(input)
-- 			-- forward
-- 			local output = net:forward(noisy_input)
-- 			local loss = loss_func:forward(output, target)

-- 			local peek_criterion = nnsparse.SparseCriterion(nn.MSECriterion())
-- 			peek_criterion.sizeAverage = false
-- 			local cnt_slot = 0
-- 			for _, item_batch in pairs(input) do
-- 				cnt_slot = cnt_slot + item_batch:size(1)
-- 			end
-- 			local rmse = math.sqrt(peek_criterion:forward(output, target) / cnt_slot)
-- 			print('current rmse: '..2.5*rmse)

-- 			-- backward
-- 			local dloss = loss_func:backward(output, target)
-- 			net:backward(noisy_input, dloss)
-- 			-- print('current loss (MSE): '..(loss/batch))
-- 			return loss / batch, dw:div(batch)
-- 		end
-- 		sgdConfiguration.evalCounter = epc
-- 		optim.sgd(feval, w, sgdConfiguration)
-- 	end
-- end


local function testAE(net)
	
	-- create minibatch
	local pred_cnt = 0
	local input, target, minibatch = {}, {}, {}

	for k, _ in pairs(train) do
		if valid[k] ~= nil and valid[k].curSize == nil then
			input[#input + 1] = train[k]
			target[#target + 1] = valid[k]
			pred_cnt = pred_cnt + valid[k]:size(1)
			
			if #input == batch then
				minibatch[#minibatch + 1] = {input = input, target = target}
				input, target = {}, {}
			end
		end
	end

	if #input > 0 then
		minibatch[#minibatch + 1] = {input = input, target = target}
		input, target = {}, {}
	end

	local criterion = nnsparse.SparseCriterion(nn.MSECriterion())
	criterion.sizeAverage = false

	-- compute RMSE

	local err = 0
	for valid_cnt, validBacth in pairs(minibatch) do
		xlua.progress(valid_cnt, #minibatch)
		local output = net:forward(validBacth.input)
		err = err + criterion:forward(output, validBacth.target)
	end
	-- print(err, pred_cnt)
	err = err / pred_cnt
	rmse = math.sqrt(err)
	print("current RMSE: "..2.5*rmse)
	return rmse, net

end

-- test network
print("Start testing...")
testAE(net)

