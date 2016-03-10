require('nn')
require('gnuplot')

DSP_LIMIT = 3600
DSP_ADD = 2
DSP_MUL = 3
Cycle_diff = 100
AllSolution = 1

function execution_cycles(layer, tuple)
 local cycles = math.ceil(layer['M']/tuple['Tm']) * math.ceil(layer['N']/tuple['Tn']) * math.ceil(layer['R']/tuple['Tr']) * math.ceil(layer['C']/tuple['Tc']) *  layer['K'] *  layer['K'] 
 return cycles   
end

function DSP(tuple)
local dsp = tuple['Tm'] * tuple['Tn'] * tuple['Tr'] * tuple['Tc'] * (DSP_ADD + DSP_MUL) 
return dsp
end

function search_options(table, key)
  local cycleL, tupleL
  local DSPL = DSP_LIMIT
  local flag = false
  for cycle, _ in pairs(table) do
    if(math.abs(cycle - key)  < Cycle_diff and table[cycle]['dsp'] < DSPL) then
	 cycleL = cycle
	 DSPL = table[cycle]['dsp']
	 tupleL = table[cycle]['tuple']
	 flag = true
	end
  end

  return cycleL, DSPL, tupleL, flag    
end

function TimeMultiplex(pyramid, layerT, outfile)

	local table_dsp = {}
	local table_cycle = {}
	local N = #layerT 
	local table_layer = {}

	for i = 1, #layerT do
	layer = layerT[i]
	table_layer[i] = {}
	table_dsp[i] = {}
	table_cycle[i] = {}
		for tm = 1, layer['M'], 1 do
		   for tn = 1, layer['N'], 1 do
		    for tr = 1, layer['R'], 1 do
		     for tc = 1, layer['C'], 1  do
			 count = count + 1
			 --print(count)
		     local tuple = {}
			 tuple['Tm'] = tm
			 tuple['Tn'] = tn
			 tuple['Tr'] = tr
			 tuple['Tc'] = tc
			 local cycles = execution_cycles(layer, tuple)
			 local dsp = DSP(tuple)
			 tab = {}
			 tab['dsp'] = dsp
			 tab['tuple'] = tuple
	
			 if(dsp <= DSP_LIMIT) then
				if(table_layer[i][cycles] == nil) then			  
				  table_layer[i][cycles] = tab		    		 
	  			else
				  if(table_layer[i][cycles]['dsp'] > dsp) then
					table_layer[i][cycles] = nil
					table_layer[i][cycles] = tab
				  end				
				end
				table.insert(table_cycle[i], cycles)
				table.insert(table_dsp[i], dsp)         			
			end		 

	 	  end
		end
	  end
	 end
	end

local match_cycle = {}
local match_dsp = {}
local match_tuple = {}

local match_cycleG = {}
local match_dspG = {}

local count = 0
for cycle, _ in pairs(table_layer[1]) do
   local tuple = table_layer[1][cycle]['tuple']
   local dsp = table_layer[1][cycle]['dsp']
   local dsp_t = dsp
  
   local cycle_layer = {}
   local dsp_layer = {}
   local tuple_layer = {}

   cycle_layer[1] = cycle
   dsp_layer[1] = dsp
   tuple_layer[1] = tuple 

   for i = 2, N do
	 local cycleL, DSPL, tupleL, flag = search_options(table_layer[i], cycle)
	 if(flag == false or dsp_t + DSPL > DSP_LIMIT) then
		break
	 end

	dsp_t = dsp_t + DSPL		
	cycle_layer[i] = cycleL
	dsp_layer[i] = DSPL
	tuple_layer[i] = tupleL
	count = count + 1
	if(i == N) then
	  match_cycleL = {}
	  match_dspL = {}
	  match_tupleL = {}

	  for j = 1, N do
	   table.insert(match_cycleL, cycle_layer[j])
       table.insert(match_dspL, dsp_layer[j])
	   table.insert(match_tupleL, tuple_layer[j])
	  end

	  table.insert(match_cycle, match_cycleL)
      table.insert(match_dsp, match_dspL)
	  table.insert(match_tuple, match_tupleL)

	  table.insert(match_cycleG, match_cycleL[1])
      table.insert(match_dspG, dsp_t)
	  		
	end 

  end
  
end


outfile:write(string.format("\n Different Convolution layer properties for this pyramids \n"))
outfile:write(string.format("\nResource\t\t\t"))
for j = 1, N do
  outfile:write(string.format("Layer %d \t\t", j))  
end
outfile:write(string.format("\n"))  

outfile:write(string.format("Output Maps(M) \t\t")) 
for j = 1, N do
  outfile:write(string.format("%6d \t\t\t", layerT[j]['M']))
end
outfile:write(string.format("\n"))  

outfile:write(string.format("Input Maps(N) \t\t")) 
for j = 1, N do
  outfile:write(string.format("%6d \t\t\t", layerT[j]['N']))
end
outfile:write(string.format("\n"))  

outfile:write(string.format("Rows (R) \t\t\t")) 
for j = 1, N do
  outfile:write(string.format("%6d \t\t\t", layerT[j]['R']))
end
outfile:write(string.format("\n"))  

outfile:write(string.format("Rows (C) \t\t\t")) 
for j = 1, N do
  outfile:write(string.format("%6d \t\t\t", layerT[j]['R']))
end
outfile:write(string.format("\n"))  

outfile:write(string.format("Filter (K) \t\t\t")) 
for j = 1, N do
  outfile:write(string.format("%6d \t\t\t", layerT[j]['K']))
end
outfile:write(string.format("\n"))  

outfile:write(string.format("Stride (S) \t\t\t")) 
for j = 1, N do
  outfile:write(string.format("%6d \t\t\t", layerT[j]['S']))
end
outfile:write(string.format("\n"))  


if(match_cycle[1] == nil) then
  outfile:write(string.format("No Solution possible for this case\n\n"))
else

	min_cycle = match_cycle[1][1]
	index = 1

	for i = 1, #match_cycle do
	if(AllSolution == 1) then
		 outfile:write(string.format("\nSolution %d:\n\n", i))
		 outfile:write(string.format("\nTotal DSP Units used %d\n", match_dspG[i]))
		 outfile:write(string.format("\n Resource\t\t\t"))
		 for j = 1, N do
		  outfile:write(string.format("Layer %d \t\t", j))  
		 end
		 outfile:write(string.format("\n"))  
		 outfile:write(string.format("\n Cycles\t\t\t\t"))
	end 

	 for j = 1, N do
		if(AllSolution==1) then
		  outfile:write(string.format("%6d \t\t\t", match_cycle[i][j]))
		end

	    if(match_cycle[i][j] < min_cycle) then
		  index = i
		  min_cycle = match_cycle[i][j]
	    end
	 end

	if(AllSolution==1) then
		 outfile:write(string.format("\n"))  

		 outfile:write(string.format("\n DSPs\t\t\t\t")) 
		 for j = 1, N do
		  outfile:write(string.format("%6d \t\t\t", match_dsp[i][j]))  
		 end
		 outfile:write(string.format("\n"))  
		 
		 outfile:write(string.format("\n Unroll Factor TM\t")) 
		 for j = 1, N do
		  outfile:write(string.format("%6d \t\t\t", match_tuple[i][j]['Tm']))  
		 end
		 outfile:write(string.format("\n"))  
		 
		 outfile:write(string.format("\n Unroll Factor TN\t")) 
		 for j = 1, N do
		  outfile:write(string.format("%6d \t\t\t", match_tuple[i][j]['Tn']))  
		 end
		 outfile:write(string.format("\n"))  

		 outfile:write(string.format("\n Unroll Factor TR\t")) 
		 for j = 1, N do
		  outfile:write(string.format("%6d \t\t\t", match_tuple[i][j]['Tr']))  
		 end
		 outfile:write(string.format("\n"))  

		 outfile:write(string.format("\n Unroll Factor TC\t")) 
		 for j = 1, N do
		  outfile:write(string.format("%6d \t\t\t", match_tuple[i][j]['Tc']))  
		 end

		 outfile:write(string.format("\n"))  

	end

	end    

	 --- Best Solution
	 outfile:write(string.format("\nRecommended Solution with minimum cycles and Max DSP units:\n\n"))
	 outfile:write(string.format("\nSolution %d:\n\n", index))

	 outfile:write(string.format("\nTotal DSP Units used %d\n", match_dspG[index]))
	 outfile:write(string.format("\n Resource\t\t\t"))
	 for j = 1, N do
	  outfile:write(string.format("Layer %d \t\t", j))  
	 end
	 outfile:write(string.format("\n"))  

	 outfile:write(string.format("\n Cycles \t\t\t")) 
	 for j = 1, N do
	  outfile:write(string.format("%6d \t\t\t", match_cycle[index][j]))
	 end
	 outfile:write(string.format("\n"))  

	 outfile:write(string.format("\n DSPs\t\t\t\t")) 
	 for j = 1, N do
	  outfile:write(string.format("%6d \t\t\t", match_dsp[index][j]))  
	 end
	 outfile:write(string.format("\n"))  
	 
	 outfile:write(string.format("\n Unroll Factor TM\t")) 
	 for j = 1, N do
	  outfile:write(string.format("%6d \t\t\t", match_tuple[index][j]['Tm']))  
	 end
	 outfile:write(string.format("\n"))  
	 
	 outfile:write(string.format("\n Unroll Factor TN\t")) 
	 for j = 1, N do
	  outfile:write(string.format("%6d \t\t\t", match_tuple[index][j]['Tn']))  
	 end
	 outfile:write(string.format("\n"))  

	 outfile:write(string.format("\n Unroll Factor TR\t")) 
	 for j = 1, N do
	  outfile:write(string.format("%6d \t\t\t", match_tuple[index][j]['Tr']))  
	 end
	 outfile:write(string.format("\n"))  

	 outfile:write(string.format("\n Unroll Factor TC\t")) 
	 for j = 1, N do
	  outfile:write(string.format("%6d \t\t\t", match_tuple[index][j]['Tc']))  
	 end

	 outfile:write(string.format("\n"))  

if(plot) then
	T_cycle = {}
	T_dsp = {}
	T_cycle = torch.FloatTensor(match_cycleG)
	T_dsp = torch.FloatTensor(match_dspG)

	figure = pyramid..'.png'
	gnuplot.pngfigure(figure)
	gnuplot.title('Same Execution Cycle at each layer for pyramids '.. pyramid)
	gnuplot.xlabel('DSP units used')
	gnuplot.ylabel('Execution cycles')
	gnuplot.plot(T_dsp, T_cycle, '+')
	gnuplot.plotflush()
end

end




end

--[[
--Only for middle pyramids
layer1 = {}
layer1['R'] = 4
layer1['C'] = 4
layer1['M'] = 48
layer1['N'] = 3
layer1['K'] = 11
layer1['S'] = 4


layer2 = {}
layer2['R'] = 2
layer2['C'] = 2
layer2['M'] = 128
layer2['N'] = 48
layer2['K'] = 5
layer2['S'] = 1

layerT = {}
layerT[1] = layer1
layerT[2] = layer2
cases = {}
count = 0


outfile = io.open("Same_Execution_cycles_Result", "w")
TimeMultiplex(layerT, outfile)
]]--

