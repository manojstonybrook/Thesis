require('nn')
require('gnuplot')

outfile = io.open("Same_Execution_cycles_Result", "w")
DSP_LIMIT = 2800
DSP_ADD = 2
DSP_MUL = 3
Cycle_diff = 100

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

function TimeMultiplex(layerT, outfile)

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
					print(table_layer[i][cycles])
					table_layer[i][cycles] = nil
					table_layer[i][cycles] = tab
					print(table_layer[i][cycles])
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

count = 0
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

--[[
print('match_cycle')
print(match_cycle)
print('match_dsp')
print(match_dsp)
print('match_tuple')
print(match_tuple)
]]--

for i = 1, N do
	outfile:write('layer'..i)
	outfile:write(string.format("\n"))
	outfile:write(string.format("Output Feature Maps (M) = %d \n", layerT[i]['M']))
	outfile:write(string.format("input  Feature Maps (N) = %d \n", layerT[i]['N']))
	outfile:write(string.format("         input Rows (R) = %d \n", layerT[i]['R']))
	outfile:write(string.format("         input Cols (C) = %d \n", layerT[i]['C']))

	outfile:write(string.format("\n\n"))
end

outfile:write(string.format("Different Solutions with almost same execution cycles in different layers\n\n"))

min_cycle = match_cycle[1][1]
index = 1

for i = 1, #match_cycle do

 outfile:write(string.format("\nSolution %d:\n\n", i))
 outfile:write(string.format("\nTotal DSP Units used %d\n", match_dspG[i]))
 outfile:write(string.format("\n Resource"))
 for j = 1, N do
  outfile:write(string.format("\t\t\t\t Layer %d \t\t", j))  
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n Cycles")) 
 for j = 1, N do

  outfile:write(string.format("\t\t\t\t   %6d \t\t", match_cycle[i][j]))
  if(match_cycle[i][j] < min_cycle) then
    index = i
	min_cycle = match_cycle[i][j]
  end
  
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n DSPs")) 
 for j = 1, N do
  outfile:write(string.format("\t\t\t\t   %6d \t\t", match_dsp[i][j]))  
 end
 outfile:write(string.format("\n"))  
 
 outfile:write(string.format("\n Unroll Factor TM")) 
 for j = 1, N do
  outfile:write(string.format("\t   %6d \t\t\t\t\t", match_tuple[i][j]['Tm']))  
 end
 outfile:write(string.format("\n"))  
 
 outfile:write(string.format("\n Unroll Factor TN")) 
 for j = 1, N do
  outfile:write(string.format("\t   %6d \t\t\t\t\t", match_tuple[i][j]['Tn']))  
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n Unroll Factor TR")) 
 for j = 1, N do
  outfile:write(string.format("\t   %6d \t\t\t\t\t", match_tuple[i][j]['Tr']))  
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n Unroll Factor TC")) 
 for j = 1, N do
  outfile:write(string.format("\t   %6d \t\t\t\t\t", match_tuple[i][j]['Tc']))  
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n --------------------------- \n"))

end    



 --- Best Solution
 outfile:write(string.format("\nRecommended Solution with minimum cycles and Max DSP units:\n\n"))


 outfile:write(string.format("\nSolution %d:\n\n", index))

 outfile:write(string.format("\nTotal DSP Units used %d\n", match_dspG[index]))
 outfile:write(string.format("\n Resource"))
 for j = 1, N do
  outfile:write(string.format("\t\t\t\t Layer %d \t\t", j))  
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n Cycles")) 
 for j = 1, N do
  outfile:write(string.format("\t\t\t\t   %6d \t\t", match_cycle[index][j]))
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n DSPs")) 
 for j = 1, N do
  outfile:write(string.format("\t\t\t\t   %6d \t\t", match_dsp[index][j]))  
 end
 outfile:write(string.format("\n"))  
 
 outfile:write(string.format("\n Unroll Factor TM")) 
 for j = 1, N do
  outfile:write(string.format("\t   %6d \t\t\t\t\t", match_tuple[index][j]['Tm']))  
 end
 outfile:write(string.format("\n"))  
 
 outfile:write(string.format("\n Unroll Factor TN")) 
 for j = 1, N do
  outfile:write(string.format("\t   %6d \t\t\t\t\t", match_tuple[index][j]['Tn']))  
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n Unroll Factor TR")) 
 for j = 1, N do
  outfile:write(string.format("\t   %6d \t\t\t\t\t", match_tuple[index][j]['Tr']))  
 end
 outfile:write(string.format("\n"))  

 outfile:write(string.format("\n Unroll Factor TC")) 
 for j = 1, N do
  outfile:write(string.format("\t   %6d \t\t\t\t\t", match_tuple[index][j]['Tc']))  
 end
 outfile:write(string.format("\n"))  

outfile:write(string.format("\n --------------------------- \n"))

end

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


TimeMultiplex(layerT, outfile)


--[[T_cycle = {}
T_dsp = {}
T_cycle = torch.FloatTensor(match_cycleG)
T_dsp = torch.FloatTensor(match_dspG)

figure = 'layer.png'
gnuplot.pngfigure(figure)
gnuplot.title('Same Execution Cycle at each layer')
gnuplot.xlabel('DSP units used')
gnuplot.ylabel('Execution cycles')
gnuplot.plot(T_dsp, T_cycle, '+')
gnuplot.plotflush()
]]--
