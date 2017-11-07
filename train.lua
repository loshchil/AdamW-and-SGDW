--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found here
--  https://github.com/facebook/fb.resnet.torch/blob/master/LICENSE. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Code modified for Shake-Shake by Xavier Gastaldi
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = false,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
      targetweight = opt.targetweight
   }
   self.opt = opt

   self.opt.T_cur = 0
   self.opt.EpochNext = 1 + self.opt.Te

   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch

   ------Shake-Shake------
   if self.opt.lrShape == 'multistep' then
      self.optimState.learningRate = self:learningRate(epoch)
   end
   ------Shake-Shake------

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum, lastloss = 0.0, 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   local permutate = true
   for n, sample in dataloader:run(permutate) do

     
      if self.opt.lrShape == 'cosine' then
         --self.optimState.learningRate = self:learningRateCosine(epoch, n, trainSize)
         self.optimState.learningRate = self:learningRateCosineSGDR(epoch, n, trainSize)
         local learningRateMultiplier = self.optimState.learningRate / self.opt.LR
         local weightDecayMultiplier = learningRateMultiplier

	 self.optimState.Multiplier = learningRateMultiplier
         if ((self.opt.algorithmType == 'SGDdec') or (self.opt.algorithmType == 'ADAMdec')) then
            self.optimState.weightDecaycurrent = weightDecayMultiplier * self.optimState.weightDecay
         end
         if ((self.opt.algorithmType == 'SGDW') or (self.opt.algorithmType == 'ADAMW'))  then
            -- trainSize is the number of batches
            local weightDecayNormalized = self.optimState.weightDecay / ( math.pow(trainSize * self.opt.Te, 0.5) )
            self.optimState.weightDecaycurrent = weightDecayMultiplier * weightDecayNormalized
         end
      end
     
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      if (self.opt.algorithmType == 'SGD') 	then  self.optimState.sgd_type = 0;    optim.sgd(feval, self.params, self.optimState)  end	-- original SGD
      if (self.opt.algorithmType == 'SGDdec') 	then  self.optimState.sgd_type = 1;    optim.sgd(feval, self.params, self.optimState)  end	-- SGD with decoupling
      if (self.opt.algorithmType == 'SGDW') 	then  self.optimState.sgd_type = 1;    optim.sgd(feval, self.params, self.optimState)  end	-- SGD with decoupling and normalized weight decay
      if (self.opt.algorithmType == 'ADAM') 	then  self.optimState.adam_type = 0;   optim.adam(feval, self.params, self.optimState)  end	-- original Adam
      if (self.opt.algorithmType == 'ADAMdec') 	then  self.optimState.adam_type = 1;   optim.adam(feval, self.params, self.optimState)  end	-- Adam with decoupling
      if (self.opt.algorithmType == 'ADAMW') 	then  self.optimState.adam_type = 1;   optim.adam(feval, self.params, self.optimState)  end	-- Adam with decoupling and normalized weight decay
      

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lastloss = loss
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      --print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
      --   epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))

      ------Shake-Shake------
      print((' | Epoch: [%d][%d/%d]   Time %.3f  Data %.3f  TL %1.3f  Err %1.3f  top1 %7.2f  top5 %7.2f  lr %.4f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, lossSum / N, loss, top1, top5, self.optimState.learningRate))
      ------Shake-Shake------

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   --local predfile = io.open(string.format('pred_%d_%s_%d_%d_%d.txt', self.opt.irun, self.opt.algorithmType, self.opt.Te, self.opt.widenFactor, epoch), 'w')

   self.model:evaluate()
   local permutate = false
   for n, sample in dataloader:run(permutate) do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local toutput = torch.totable( self.model.output )
      local ttarget = torch.totable( self.target )
      if (1 == 0) then
        local s = {""}
        for i=1,#toutput do
          for j=1,#toutput[i] do
            s[#s+1] = toutput[i][j]
            s[#s+1] = " "
          end
          s[#s+1] = ttarget[i]
          s[#s+1] = " "
          s[#s+1] = "\n"
        end
        s = table.concat(s)
      end

      --if ((epoch == 1) or (epoch == 10) or (epoch == 30) or (epoch == 70) or (epoch == 148) or (epoch == 149) or (epoch == 150)) then
        --predfile:write(s)
        --predfile:flush()
      --end

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   --predfile:close()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():topk(5, 2, true, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(predictions))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

local function getCudaTensorType(tensorType)
  if tensorType == 'torch.CudaHalfTensor' then
     return cutorch.createCudaHostHalfTensor()
  elseif tensorType == 'torch.CudaDoubleTensor' then
    return cutorch.createCudaHostDoubleTensor()
  else
     return cutorch.createCudaHostTensor()
  end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

------Shake-Shake------
-- Cosine function from https://github.com/gaohuang/SnapshotEnsemble
function Trainer:learningRateCosine(epoch, iter, nBatches)
   local nEpochs_sub =  torch.floor(self.opt.nEpochs / self.opt.nCycles)
   local nEpochs_last =  self.opt.nEpochs - (self.opt.nCycles - 1) * nEpochs_sub
   local nEpochs_cur = (epoch >  (self.opt.nCycles - 1) * nEpochs_sub) and nEpochs_last or nEpochs_sub
   local T_total = nEpochs_cur * nBatches
   local T_cur = ((epoch-1) % nEpochs_cur) * nBatches + iter
   return 0.5 * self.opt.LR * (1 + torch.cos(math.pi * T_cur / T_total))
end

function Trainer:learningRateCosineSGDR(epoch, iter, nBatches)
   --print((' %7.3f %7.3f %7.3f\n'):format(self.opt.T_cur, self.opt.Te, self.opt.EpochNext))
   if (self.opt.LRdec == 'false') then return self.opt.LR   end

   self.opt.T_cur = self.opt.T_cur + 1 / (self.opt.Te * nBatches)
   if (self.opt.T_cur >= 1) then
      self.opt.T_cur = 1
   end

   if ((self.opt.T_cur >= 1) and (epoch == self.opt.EpochNext))   then
      self.opt.T_cur = 0
      self.opt.Te = self.opt.Te * self.opt.Tmult
      self.opt.EpochNext = self.opt.EpochNext + self.opt.Te
   end
   --if (self.opt.irestart == 1) then
   return 0.5 * self.opt.LR * (1 + torch.cos(math.pi * self.opt.T_cur))
end
------Shake-Shake------

return M.Trainer
