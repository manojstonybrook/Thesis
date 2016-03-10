require('nn')
require('gnuplot')
cmd = torch.CmdLine()
cmd:text()
cmd:text('Layer Parameters:')
cmd:option('--R', '', 'input model file')
cmd:option('--C', '', 'input image Width')
cmd:option('--M', '', 'input image height')
cmd:option('--N', '', 'input image channels')
cmd:option('--K', '', 'Recomp or Reuse')
cmd:option('--S', '', 'outfile')
cmd:text()
opt = cmd:parse(arg or {})



BandWidth = 4.5
Freq = 0.1 --1Ghz
BRAM_Capacity = 37080/(8*1024)
PE = 500

function execution_cycles(layer, tuple)
 local cycles = math.ceil(layer['M']/tuple['Tm']) * math.ceil(layer['N']/tuple['Tn']) * layer['R'] * layer['C'] *  layer['K'] *  layer['K'] 
 return cycles   
end

function computation_roof(layer, tuple)
  local cycles = math.ceil(layer['M']/tuple['Tm']) * math.ceil(layer['N']/tuple['Tn']) * layer['R'] * layer['C'] *  layer['K'] *  layer['K'] 
  local operations = 2 * layer['R'] * layer['C'] * layer['M'] * layer['N'] * layer['K'] * layer['K']  
  return (operations/cycles) * Freq 
end


function Computation_to_Communication_Ratio(layer, tuple)
    
  local Bweight = tuple['Tm'] * tuple['Tn'] * layer['K'] * layer['K'] * 4 
  local Bout = tuple['Tm'] * tuple['Tr'] * tuple['Tc'] * 4
  local Bin  = tuple['Tn'] * (layer['S'] * tuple['Tr'] + layer['K'] - layer['S'] ) * (layer['S'] * tuple['Tc'] + layer['K'] - layer['S'] ) * 4
  local CTC
  --print((Bin + Bout + Bweight)/(1024*1024))
  if(((Bin + Bout + Bweight)/(1024*1024)) <= BRAM_Capacity) then
    local Ain =  math.ceil(layer['M']/tuple['Tm']) * math.ceil(layer['N']/tuple['Tn']) * math.ceil(layer['R']/tuple['Tr']) * math.ceil(layer['C']/tuple['Tc']) 
    local Aweight = Ain
    local Aout = math.ceil(layer['M']/tuple['Tm']) * math.ceil(layer['R']/tuple['Tr']) * math.ceil(layer['C']/tuple['Tc']) 
  
    local operations =  2 * layer['R'] * layer['C'] * layer['M'] * layer['N'] * layer['K'] * layer['K']
  
    local external_data_access = Ain * Bin + Aweight * Bweight + Aout * Bout 
    CTC = operations/external_data_access
  end
  return CTC
end

function roofline(layer, tuple)
  local roof = computation_roof(layer, tuple)
  local CTC = Computation_to_Communication_Ratio(layer, tuple)
  local atn_perf = math.min(math.min(roof, CTC * BandWidth)) 
  return atn_perf, CTC, roof
end

layer = {}
layer['N'] = opt.N
layer['M'] = opt.M
layer['R'] = opt.R
layer['C'] = opt.C
layer['K'] = opt.K
layer['S'] = opt.S
maxCTC = 0
maxR = 0
maxC = 0
maxM = 0
maxN = 0
table_CTC = {}
table_perf = {}
--tm = 64 
--tn = 7

for tm = 1, layer['M'], 1 do
  for tn = 1, layer['N'], 1 do
    for tr = 1, layer['R'], 1 do
      for tc = 1, layer['C'], 1  do
         local tuple = {}
		 tuple['Tm'] = tm
		 tuple['Tn'] = tn
		 tuple['Tr'] = tr
		 tuple['Tc'] = tc
         if((tm * tn) < PE) then
           local perf, ctc, roof = roofline(layer, tuple)
           --print(perf, ctc, roof)   
           if(ctc) then
			 if(ctc > maxCTC) then
 				maxCTC = ctc
				maxR = tr
				maxC = tc
				maxM = tm
				maxN = tn				
              end
               
             table.insert(table_CTC, ctc)
             table.insert(table_perf, perf)
           end
         end
      end
    end
  end
end

T_perf = torch.FloatTensor(table_perf)
T_CTC = torch.FloatTensor(table_CTC)
gnuplot.pngfigure('layer2_all.png')
gnuplot.title('Roofline Model')
gnuplot.xlabel('Computation to Communication Ratio (FLOP/byte)')
gnuplot.ylabel('Attainable performance (GFLOPS)')
gnuplot.plot(T_CTC, T_perf,'+')
gnuplot.plotflush()
print('max Value', 'maxCTC=', maxCTC, 'R=', maxR, 'C=', maxC, 'M=', maxM, 'N=', maxN)

