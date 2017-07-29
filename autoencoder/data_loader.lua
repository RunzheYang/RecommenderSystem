require "torch"
require "nnsparse"

-- Meta Class
Dataset = {trainV = {}, validV = {}, trainU = {}, validU = {}}

function Dataset:new(obj, dir, train_ratio)
	obj = obj or {}
	setmetatable(obj, self)
	self.__index = self
	self.trainV, self.validV, self.trainU, self.validU = self:load_data(dir)
	-- self.bias = self:unbias()

	return Dataset
end

function Dataset:load_data(dir)
	local trainV, validV, trainU, validU = {}, {}, {}, {}
	local trainFile = io.open(dir, "r")
	print("Start loading data")
	cnt, cnt_train, cnt_valid = 0, 0, 0

	for line in trainFile:lines() do
		local userIdStr, itemIdStr, weekStr, timeStr, feat1Str, feat2Str, ratingStr 
				= line:match('(%d+)\t(%d+)\t(%d+)\t(%d+)\t(%d+)\t(%d+)\t(%d%.?%d?)')
		local userId = tonumber(userIdStr)+1
		local itemId = tonumber(itemIdStr)+1
		local week = tonumber(weekStr)
		local time = tonumber(timeStr)
		local feat1 = tonumber(feat1Str)
		local feat2 = tonumber(feat2Str)
		local rating = tonumber(ratingStr)

		-- normalize the rating between interval [-1, 1]
		rating = (rating-2.5)/2.5
		
		if torch.uniform() < train_ratio then
			-- prepare item-based training set
			trainV[itemId] = trainV[itemId] or nnsparse.DynamicSparseTensor()
			trainV[itemId]:append(torch.Tensor{userId, rating})
			-- prepare user-based training set
			trainU[userId] = trainU[userId] or nnsparse.DynamicSparseTensor()
			trainU[userId]:append(torch.Tensor{itemId, rating})
			cnt_train = cnt_train + 1
		else
			-- prepare item-based validation set
			validV[itemId] = validV[itemId] or nnsparse.DynamicSparseTensor()
			validV[itemId]:append(torch.Tensor{userId, rating})
			-- prepare user-based validation set
			validU[userId] = validU[userId] or nnsparse.DynamicSparseTensor()
			validU[userId]:append(torch.Tensor{itemId, rating})
			cnt_valid = cnt_valid + 1
		end

		if cnt % 10000 == 9999 then xlua.progress(cnt+1, 5974450) end

		cnt = cnt + 1
		-- if cnt > 100000 then break end
	end

	for k, train_item in pairs(trainV) do trainV[k] = train_item:build():ssortByIndex() end
	for k, valid_item in pairs(validV) do validV[k] = valid_item:build():ssortByIndex() end

	for k, train_item in pairs(trainU) do trainU[k] = train_item:build():ssortByIndex() end
	for k, valid_item in pairs(validU) do validU[k] = valid_item:build():ssortByIndex() end

	print("Finish data loading!")
	print(cnt_train, "records are in training set")
	print(cnt_valid, "records are in validation set")

	return trainV, validV, trainU, validU
end

-- function Dataset:unbias()
-- 	bias = {}
-- 	for k, train_item in pairs(self.train) do
-- 		local mean = train_item[{{}, 2}]:mean()
-- 		self.train[k][{{}, 2}]:add(-mean)
		
-- 		if self.valid[k] ~= nil and self.valid[k].curSize == nil then
-- 			self.valid[k][{{}, 2}]:add(-mean)
-- 		end
-- 		bias[k] = mean
-- 		if k % 1000 == 999 then xlua.progress(k, 99782) end
-- 	end
-- 	return bias
-- end

return Dataset

-- dataset = Dataset:new(nil, "item_recom/train_info.tsv", 0.9)
