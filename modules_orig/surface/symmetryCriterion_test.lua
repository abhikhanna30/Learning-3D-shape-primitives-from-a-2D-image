require 'optim'
require 'cunn'
local M = {}

-------------------------------
-------------------------------
local function addTableRec(t1,t2,w1,w2)
    if(torch.isTensor(t1)) then
        return w1*t1+w2*t2
    end
    local result = {}
    for ix=1,#t1 do
        result[ix] = addTableRec(t1[ix],t2[ix],w1,w2)
    end
    return result
end

local SymmetryCriterion = {}
SymmetryCriterion.__index = SymmetryCriterion

setmetatable(SymmetryCriterion, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function SymmetryCriterion.new(pointSamplerModule, tsdfModule, nSamples)
    local self = setmetatable({}, SymmetryCriterion)
    self.pointSamplerModule = pointSamplerModule
    self.tsdfModule = tsdfModule
    self.nSamples = nSamples
 
    --
    local impWeights = - nn.Identity()
    local totWeights = impWeights - nn.Sum(2) - nn.AddConstant(1e-6) - nn.Replicate(nSamples,2) --constant handles cases where all parts are null parts
    local normWeights = {impWeights, totWeights} - nn.CDivTable()
    self.normWeightsModule = nn.gModule({impWeights}, {normWeights})
    
    return self
end

function SymmetryCriterion:cuda()
    self.useGpu = true
    self.pointSamplerModule = self.pointSamplerModule:cuda()
    self.tsdfModule = self.tsdfModule:cuda()
    self.normWeightsModule = self.normWeightsModule:cuda()
end

function SymmetryCriterion:forward(input)
    -- Loss is 1/batchSize * tsdfLoss * normWeights
    self.samples, self.impWeights = unpack(self.pointSamplerModule:forward(input))
    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symsamples.dat', self.samples)
    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symimpWeights.dat', self.impWeights)
    
    self.samplesRef = self.samples:clone()
    self.samplesRef:narrow(3,3,1):copy(-1*self.samples:narrow(3,3,1))
    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symsamplesRef.dat', self.samplesRef)
    
    self.tsdfLosses = self.tsdfModule:forward({self.samplesRef,input})

    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symtsdfLosses.dat', self.tsdfLosses)
    self.normWeights = self.normWeightsModule:forward(self.impWeights)

    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symnormWeights.dat', self.normWeights)
    --print(self.normWeights:sum(2):mean(1))
    
    self.sampleWeights = self.normWeights:clone():expand(self.samples:size())
    self.weightedLosses = self.tsdfLosses:clone():cmul(self.normWeights)

    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symsampleWeights.dat', self.sampleWeights)

    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symweightedLosses.dat', self.weightedLosses)
    
    return self.weightedLosses:sum(2):mean()
end

function SymmetryCriterion:backward(input)
    local normalizationFactor = (1/self.samples:size(1))
    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symnormalizationFactor.dat', normalizationFactor)
    local gradNorm = self.tsdfLosses*normalizationFactor


    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symgradNorm.dat', gradNorm)
    local gradImp = self.normWeightsModule:updateGradInput(self.impWeights, gradNorm)
    --gradImp:fill(0)
    
    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symgradImp.dat', gradImp)

    local gradLossTsdf = normalizationFactor*self.normWeights

    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symgradLossTsdf.dat', gradLossTsdf)
    local gradTsdf = self.tsdfModule:backward({self.samplesRef,input}, gradLossTsdf:squeeze())
    
    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symgradTsdf.dat', gradTsdf)
    
    local gradPointsRef = gradTsdf[1]
    local gradPoints = gradTsdf[1]:clone()
    gradPoints:narrow(3,3,1):copy(-1*gradPointsRef:narrow(3,3,1))
    
    --print(self.samples[1], self.impWeights[1], self.normWeights[1],gradSamples[1], gradImp[1])
    
    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symgradPoints.dat', gradPoints)
    local gradInput_s = self.pointSamplerModule:updateGradInput(input, {gradPoints, gradImp})
    self.gradInput = addTableRec(gradInput_s, gradTsdf[2], 1, 1)
    
    
    torch.save('/home/abhijeek/volumetricPrimitives/cachedir/snapshots/St2_chairScene_ChamferSurf_null_small_ft_Part10_prob0pt2_shape0pt5_null8em5/symgradInput.dat', self.gradInput)
    
    return self.gradInput
end
-------------------------------
-------------------------------
M.SymmetryCriterion = SymmetryCriterion
return M
