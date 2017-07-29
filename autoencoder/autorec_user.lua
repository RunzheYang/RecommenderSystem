require 'torch'
require 'nn'
require 'optim'
require 'nnsparse'
require 'xlua'
require 'data_loader'

local item_num = 99782
local topic = 64
local batch = 256
local epoch = 30

local sgdConfiguration = {
	learningRate      = 0.9, 
	learningRateDecay = 0.002,
	weightDecay       = 0.0005,
}

-- load data
local dataset = Dataset:new(nil, "../item_recom/train_info.tsv", 0.9)
local train = dataset.trainU
local valid = dataset.validU

-- build network
local net = nn.Sequential()
net:add(nnsparse.SparseLinearBatch(item_num, topic))
net:add(nn.Tanh())
net:add(nn.Linear(topic, item_num))
net:add(nn.Tanh())

-- net = torch.load('models/ae_simple_64.models')

print(net)

-- define the loss function
-- local loss_func = nnsparse.SDAESparseCriterion(nn.MSECriterion(), {
local loss_func = nnsparse.MaskCriterion(nn.MSECriterion(), {
		alpha = 1,
		beta = 0.48,
		hideRatio = 0.38,
	})
loss_func.inputDim = item_num
loss_func.sizeAverage = false

local function batchify(data)
	local input, minibatch = {}, {}
	local shuffle = torch.randperm(item_num)
	shuffle:apply(function(k) 
			if data[k] then
				input[#input + 1] = data[k]
				if #input == batch then
					minibatch[#minibatch + 1] = input
					input = {}
				end
			end
		end)
	if #input > 0 then
		minibatch[#minibatch + 1] = input
		input = {}
	end
	return minibatch
end


local function trainAE(net, epc)

	-- create minibatch
	local input, minibatch = {}, batchify(train)

	local w, dw = net:getParameters()
	for cnt, input in pairs(minibatch) do
		xlua.progress(cnt, #minibatch)
		local function feval(x)
			-- reset gradients and losses
			net:zeroGradParameters()
			-- set target as input
			local target = input
			-- add noise to input
			local noisy_input = loss_func:prepareInput(input)
			-- forward
			local output = net:forward(noisy_input)
			local loss = loss_func:forward(output, target)

			-- local peek_criterion = nnsparse.SparseCriterion(nn.MSECriterion())
			-- peek_criterion.sizeAverage = false
			-- local cnt_slot = 0
			-- for _, item_batch in pairs(input) do
			-- 	cnt_slot = cnt_slot + item_batch:size(1)
			-- end
			-- local rmse = math.sqrt(peek_criterion:forward(output, target) / cnt_slot)
			-- print('current rmse: '..2.5*rmse)

			-- backward
			local dloss = loss_func:backward(output, target)
			dloss = dloss
			net:backward(noisy_input, dloss)
			-- print('current loss (MSE): '..(loss/batch))
			return loss / batch, dw:div(batch)
		end
		sgdConfiguration.evalCounter = epc
		optim.sgd(feval, w, sgdConfiguration)
	end
end


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
	return 2.5*rmse, net

end

local best_rmse, _ = testAE(net)
print("starting rmse "..best_rmse)
-- best_rmse = 1.442
-- train network
print("Start training...")
for epc = 1, epoch do
	xlua.progress(epc, epoch)
	trainAE(net, epc)
	rmse, finalnet = testAE(net)
	if rmse < best_rmse then
		best_rmse = rmse
		torch.save('models/ae_user_64.models', finalnet)
	end
end
print("Best RMSE: "..best_rmse)

