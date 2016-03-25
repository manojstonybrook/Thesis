require('nn')

function FPGAParameters(Network, bottom, top, outfile)

    local pixel1 = {}
    local pixel2 = {}
    
    local tab = {}
    tab['tl'] = {0, 0}
    tab['br'] = {0, 0}
    pixel1[bottom+1] = tab
    
    tab = {}
    tab['tl'] = {0, 1}
    tab['br'] = {0, 1}
    pixel2[bottom+1] = tab
    
    outfile:write(string.format("%10s - %s Pyramid \n\n", top-1, bottom))
    
    --print(Network)
    if(top == 0) then
      top = 1
    end 
    
    pixel1 = calculate_roi(Network, bottom, pixel1, nil, top)
    pixel2 = calculate_roi(Network, bottom, pixel2, nil, top)
    
    local stride1 = pixel2[top]['br'][2] 
    local stride2 = pixel1[top]['br'][2]
    local stride = stride1 - stride2
    
    outfile:write(string.format("const int strideX = %d; \nconst int strideY = %d;\n", stride, stride))
    
    local countConv = 1
    local countPool = 1
    local countPad = 1

    -- initializing parameters
    for i = top, bottom+1 do
       if(i == 1) then
         outfile:write(string.format("const int inputX = %d;\nconst int inputY = %d;\n", pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
       else if(Network.modules[i-1].__typename == 'nn.SpatialConvolution') then   
           outfile:write(string.format("const int f%dK =  %d;\nconst int f%dS =  %d;\n", countConv, Network.modules[i-1].kH, countConv, Network.modules[i-1].dH))
           outfile:write(string.format("const int TM%d =  %d;\nconst int TN%d =  %d;\n", countConv, Network.modules[i-1].nOutputPlane, countConv, Network.modules[i-1].nInputPlane)) 
           outfile:write(string.format("const float filter%d[%d][%d][%d][%d];\n\n", countConv, Network.modules[i-1].nOutputPlane, Network.modules[i-1].nInputPlane, Network.modules[i-1].kH, Network.modules[i-1].kW))

		   if(countConv > 1) then
			  outfile:write(string.format("const int HR%d =  %d, WR%d =  f%dK - f%dS;\nconst int HB%d =  f%dK - f%dS, WB%d =  %d;\n", countConv, pixel1[i-1]['br'][1]-pixel1[i-1]['tl'][1]+1, countConv, countConv, countConv, countConv, countConv, countConv, countConv,Network.modules[i-1].output:size()[3]))
           end

		   countConv = countConv + 1 
       else if(Network.modules[i-1].__typename == 'nn.SpatialMaxPooling') then   
           outfile:write(string.format("const int pool%dK = %d;\nconst int pool%dS = %d;\n\n", countPool, Network.modules[i-1].kH, countPool, Network.modules[i-1].dH))
           countPool = countPool + 1
       else if(Network.modules[i-1].__typename == 'nn.SpatialZeroPadding') then   
           outfile:write(string.format("const float pad%d = %d\n\n", countPad, Network.modules[i-1].pad_l)) 
           countPad = countPad + 1       
       else
       
       end
       end
       end
       end   
   end

    
    countConv = 1
    countPool = 1
    countPad = 1
    
    --initializing on-chip buffers
    for i = top, bottom do
       if(i == 1) then
         outfile:write(string.format("\n\nfloat inputP[%d][%d][%d]\n", opt.channels, pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
       else if(Network.modules[i-1].__typename == 'nn.SpatialConvolution') then

			if (countConv > 1) then
			  outfile:write(string.format("float onchipB%d[%d][%d][%d]\n", countConv, Network.modules[i-2].output:size()[1], Network.modules[i-1].kH - Network.modules[i-1].dH, Network.modules[i-2].output:size()[2]))
			  outfile:write(string.format("float onchipR%d[%d][%d][%d]\n", countConv, Network.modules[i-2].output:size()[1], pixel1[i-1]['br'][1]-pixel1[i-1]['tl'][1]+1, Network.modules[i-1].kH - Network.modules[i-1].dH))
           end             
      
           outfile:write(string.format("float conv%d[%d][%d][%d];\n", countConv, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
           countConv = countConv + 1 
       else if(Network.modules[i-1].__typename == 'nn.SpatialMaxPooling') then   
           outfile:write(string.format("float pool%d[%d][%d][%d];\n", countPool, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1))
           countPool = countPool + 1
       else if(Network.modules[i-1].__typename == 'nn.SpatialZeroPadding') then   
           outfile:write(string.format("float pad%d[%d][%d][%d];\n", countPad, Network.modules[i-1].output:size()[1], pixel1[i]['br'][1]-pixel1[i]['tl'][1]+1, pixel1[i]['br'][2]-pixel1[i]['tl'][2]+1)) 
           countPad = countPad + 1       
       else
       
       end
       end
       end
       end   
   end
   
   --calc params
    countConv = 1
    countPool = 1
    countPad = 1
    --parameters used by network
    for i = top, bottom do
       if(i == top) then
           outfile:write(string.format("int inputW, inputH;\n"))
       end    
       if(Network.modules[i].__typename == 'nn.SpatialConvolution') then
           outfile:write(string.format("int inConv%dW, inConv%dH, outConv%dW, outConv%dH;\n", countConv, countConv, countConv, countConv));
           countConv = countConv + 1 
       else if(Network.modules[i].__typename == 'nn.SpatialMaxPooling') then   
           outfile:write(string.format("int inPool%dW, inPool%dH, outPool%dW, outPool%dH;\n", countPool, countPool, countPool, countPool));
           countPool = countPool + 1
       else if(Network.modules[i].__typename == 'nn.SpatialZeroPadding') then   
           outfile:write(string.format("int inPad%dW, inPad%dH, outPad%dW, outPad%dH;\n", countPad, countPad, countPad, countPad));
           outfile:write(string.format("int pad%d_l, pad%d_r, pad%d_t, pad%d_b;\n", countPad, countPad, countPad, countPad));           
           countPad = countPad + 1       
       end
       end
       end   
   end
   
   
   -- Number of groups Calculation
   local poolsize = {}
   local count = 1
   for i = top, bottom do
      if(Network.modules[i].__typename == 'nn.SpatialConvolution' or Network.modules[i].__typename == 'nn.SpatialMaxPooling') then
         local First = pixel1[i+1]['br'][2]-pixel1[i+1]['tl'][2]+1
         local Next = pixel2[i+1]['br'][2]-pixel1[i+1]['br'][2]
         local Total = Network.modules[i].output:size()[3]
         local tab = {}
         tab['layer'] = i
         tab['col'] = math.ceil((Total - First)/Next) + 1
         poolsize[count]  = tab
         count = count + 1
        
      end
   end
   
   
   local Ngroup = 1
   local Group = {}
   local start = poolsize[1]['col']
   for i = 2, #poolsize do 
     if (poolsize[i]['col'] ~= start) then
        tab = {}
        tab['layer'] = poolsize[i-1]['layer']
        tab['cond'] = start
        if(Network.modules[tab['layer']].__typename == 'nn.SpatialConvolution') then -- taking nonlinearlty into account 
           tab['layer'] = tab['layer'] + 1  
        end
        Group[Ngroup] = tab
        Ngroup = Ngroup + 1
        start = poolsize[i]['col']
     end 
   end
   
   Ngroup = #Group
   for i = 1, Ngroup do
      outfile:write(string.format("bool Group%d;\n", i))
      local cond = Group[i]['cond']
      outfile:write(string.format("int Group%dcond = %d;\n", i, cond))
  end 
  outfile:write(string.format("bool Group%d;\n", #Group+1))
  outfile:write(string.format("int Group%dcond = %d;\n", #Group+1, Network.modules[bottom].output:size()[3]))
  
  
  
  -----------------------------------------------------------------------------------------------
    -- Code for row, col loop
    
    outfile:write(string.format("const int NRows = %d;\nconst int NCols = %d;\n", Network.modules[bottom].output:size()[2], Network.modules[bottom].output:size()[3]))  
    outfile:write(string.format("int rowT;\nint colT;\n\n"))  
    
    
    
    outfile:write(string.format("for(int row = 0; row < NRows; row++){\n"))
    outfile:write(string.format("\tfor(int col = 0; col < NCols; row++){\n\n"))
    local ifcond = true
    local GN = 1
    
    outfile:write(string.format("\t\trowT = inputY + (row-1) * strideY + f1K - f1S;\n" ))  
    outfile:write(string.format("\t\tcolT = inputX + (col-1) * strideX + f1K - f1S;\n\n"))  
    for i = 1, #Group do
      outfile:write(string.format("\t\tif(row == Group%dcond || col == Group%dcond)\n", i, i))
      outfile:write(string.format("\t\t Group%d = false;\n", i))
      outfile:write(string.format("\t\telse\n"))
      outfile:write(string.format("\t\t Group%d = true;\n", i))      
    end
    
    --calc parameters
    outfile:write(string.format("\t\t //calc parameters\n")) -- comment in C
    --code for row == 0 && col ==0
    --
    countConv = 1
    countPool = 1
    countPad = 1
    local prevW = "inputW" 
    local prevH = "inputH"
    outfile:write(string.format("\t\tif(row == 0 && col == 0){\n"))
    for i = top, bottom do
        if(i == top) then
           outfile:write(string.format("\t\t\tinputW = inputX;\n"))
           outfile:write(string.format("\t\t\tinputH = inputY;\n"))
        end
        
        if(Network.modules[i].__typename == 'nn.SpatialConvolution') then
            outfile:write(string.format("\n\t\t\tinConv%dW = %s;\n", countConv, prevW))
            outfile:write(string.format("\t\t\tinConv%dH = %s;\n", countConv, prevH))
            outfile:write(string.format("\t\t\toutConv%dW = (inConv%dW - f%dK)/f%dS + 1;\n", countConv, countConv, countConv, countConv))
            outfile:write(string.format("\t\t\toutConv%dH = (inConv%dH - f%dK)/f%dS + 1;\n", countConv, countConv, countConv, countConv))
            prevW = string.format("outConv%dW", countConv)
            prevH = string.format("outConv%dH", countConv)
            countConv = countConv + 1 
        else if(Network.modules[i].__typename == 'nn.SpatialMaxPooling') then 
            outfile:write(string.format("\n\t\t\tinPool%dW = %s;\n", countPool, prevW))
            outfile:write(string.format("\t\t\tinPool%dH = %s;\n", countPool, prevH))
            outfile:write(string.format("\t\t\toutPool%dW = (inPool%dW - pool%dK)/pool%dS + 1;\n", countPool, countPool, countPool, countPool))
            outfile:write(string.format("\t\t\toutPool%dH = (inPool%dH - pool%dK)/pool%dS + 1;\n", countPool, countPool, countPool, countPool))
            prevW = string.format("outPad%dW", countPool)
            prevH = string.format("outPad%dH", countPool)
            countPool = countPool + 1 
        else if(Network.modules[i].__typename == 'nn.SpatialZeroPadding') then
            outfile:write(string.format("\n\t\t\tpad%d_l = pad%d;\n", countPad, countPad))
            outfile:write(string.format("\t\t\tpad%d_r = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_t = pad%d;\n", countPad, countPad))
            outfile:write(string.format("\t\t\tpad%d_b = 0;\n", countPad))
            
            outfile:write(string.format("\t\t\tinPad%dW = %s;\n", countPad, prevW))
            outfile:write(string.format("\t\t\tinPad%dH = %s;\n", countPad, prevH))
            outfile:write(string.format("\t\t\toutPad%dW = inPad%dW + pad%d_l + pad%d_r;\n", countPad, countPad, countPad, countPad))
            outfile:write(string.format("\t\t\toutPad%dH = inPad%dH + pad%d_t + pad%d_b;\n", countPad, countPad, countPad, countPad))
            prevW = string.format("outPad%dW", countPad)
            prevH = string.format("outPad%dH", countPad)
            countPad = countPad + 1
        else
        
        end 
        end
        end
        
    end
    outfile:write(string.format("\t\t}\n"))
    
    --code for row == 0---------
    countConv = 1
    countPool = 1
    countPad = 1
    local countGroup = 1
    local prevW = "inputW" 
    local prevH = "inputH"
    
    outfile:write(string.format("\t\telse if(row == 0){\n"))
    
    for i = top, bottom do
        if(i == top) then
           outfile:write(string.format("\t\t\tinputW = strideX + f1K-f1S;\n"))
           outfile:write(string.format("\t\t\tinputH = inputY;\n"))
        end
        
        if(Network.modules[i].__typename == 'nn.SpatialConvolution') then
            outfile:write(string.format("\n\t\t\tinConv%dW = %s;\n", countConv, prevW))
            outfile:write(string.format("\t\t\tinConv%dH = %s;\n", countConv, prevH))
            outfile:write(string.format("\t\t\toutConv%dW = (inConv%dW - f%dK)/f%dS + 1;\n", countConv, countConv, countConv, countConv))
            outfile:write(string.format("\t\t\toutConv%dH = (inConv%dH - f%dK)/f%dS + 1;\n", countConv, countConv, countConv, countConv))
            prevW = string.format("outConv%dW", countConv)
            prevH = string.format("outConv%dH", countConv)
            countConv = countConv + 1 
        else if(Network.modules[i].__typename == 'nn.SpatialMaxPooling') then 
            outfile:write(string.format("\n\t\t\tinPool%dW = %s;\n", countPool, prevW))
            outfile:write(string.format("\t\t\tinPool%dH = %s;\n", countPool, prevH))
            outfile:write(string.format("\t\t\toutPool%dW = (inPool%dW - pool%dK)/pool%dS + 1;\n", countPool, countPool, countPool, countPool))
            outfile:write(string.format("\t\t\toutPool%dH = (inPool%dH - pool%dK)/pool%dS + 1;\n", countPool, countPool, countPool, countPool))
            prevW = string.format("outPad%dW", countPool)
            prevH = string.format("outPad%dH", countPool)
            countPool = countPool + 1 
        else if(Network.modules[i].__typename == 'nn.SpatialZeroPadding') then
             if(i > Group[countGroup]['layer']) then
                countGroup = countGroup + 1
            end 
            outfile:write(string.format("\n\t\t\tpad%d_l = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_r = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_t = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_b = 0;\n", countPad))
            
            outfile:write(string.format("\t\t\tif(col == Group%dcond-1)\n", countGroup, countGroup))
            outfile:write(string.format("\t\t\t\tpad%d_r = pad%d;\n", countPad, countPad))
            
            outfile:write(string.format("\t\t\tinPad%dW = %s;\n", countPad, prevW))
            outfile:write(string.format("\t\t\tinPad%dH = %s;\n", countPad, prevH))
            outfile:write(string.format("\t\t\toutPad%dW = inPad%dW + pad%d_l + pad%d_r;\n", countPad, countPad, countPad, countPad))
            outfile:write(string.format("\t\t\toutPad%dH = inPad%dH + pad%d_t + pad%d_b;\n", countPad, countPad, countPad, countPad))
            prevW = string.format("outPad%dW", countPad)
            prevH = string.format("outPad%dH", countPad)
            countPad = countPad + 1
              
        else
        
        end 
        end
        end
        
    end
    
    
    outfile:write(string.format("\t\t}\n"))
    ----------------------------------------------------
    
    countConv = 1
    countPool = 1
    countPad = 1
    countGroup = 1
    local prevW = "inputW" 
    local prevH = "inputH"
    
    outfile:write(string.format("\t\telse if(col == 0){\n"))
    
    for i = top, bottom do
        if(i == top) then
           outfile:write(string.format("\t\t\tinputW = inputX;\n"))
           outfile:write(string.format("\t\t\tinputH = strideY + f1K-f1S;\n"))
        end
        
        if(Network.modules[i].__typename == 'nn.SpatialConvolution') then
            outfile:write(string.format("\n\t\t\tinConv%dW = %s;\n", countConv, prevW))
            outfile:write(string.format("\t\t\tinConv%dH = %s;\n", countConv, prevH))
            outfile:write(string.format("\t\t\toutConv%dW = (inConv%dW - f%dK)/f%dS + 1;\n", countConv, countConv, countConv, countConv))
            outfile:write(string.format("\t\t\toutConv%dH = (inConv%dH - f%dK)/f%dS + 1;\n", countConv, countConv, countConv, countConv))
            prevW = string.format("outConv%dW", countConv)
            prevH = string.format("outConv%dH", countConv)
            countConv = countConv + 1 
        else if(Network.modules[i].__typename == 'nn.SpatialMaxPooling') then 
            outfile:write(string.format("\n\t\t\tinPool%dW = %s;\n", countPool, prevW))
            outfile:write(string.format("\t\t\tinPool%dH = %s;\n", countPool, prevH))
            outfile:write(string.format("\t\t\toutPool%dW = (inPool%dW - pool%dK)/pool%dS + 1;\n", countPool, countPool, countPool, countPool))
            outfile:write(string.format("\t\t\toutPool%dH = (inPool%dH - pool%dK)/pool%dS + 1;\n", countPool, countPool, countPool, countPool))
            prevW = string.format("outPad%dW", countPool)
            prevH = string.format("outPad%dH", countPool)
            countPool = countPool + 1 
        else if(Network.modules[i].__typename == 'nn.SpatialZeroPadding') then
             if(i > Group[countGroup]['layer']) then
                countGroup = countGroup + 1
            end 
            outfile:write(string.format("\n\t\t\tpad%d_l = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_r = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_t = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_b = 0;\n", countPad))
            
            
            outfile:write(string.format("\t\t\tif(row == Group%dcond-1)\n", countGroup, countGroup))
            outfile:write(string.format("\t\t\t\tpad%d_b = pad%d;\n", countPad, countPad))
            
            
            outfile:write(string.format("\t\t\tinPad%dW = %s;\n", countPad, prevW))
            outfile:write(string.format("\t\t\tinPad%dH = %s;\n", countPad, prevH))
            outfile:write(string.format("\t\t\toutPad%dW = inPad%dW + pad%d_l + pad%d_r;\n", countPad, countPad, countPad, countPad))
            outfile:write(string.format("\t\t\toutPad%dH = inPad%dH + pad%d_t + pad%d_b;\n", countPad, countPad, countPad, countPad))
            prevW = string.format("outPad%dW", countPad)
            prevH = string.format("outPad%dH", countPad)
            countPad = countPad + 1
              
        else
        
        end 
        end
        end
        
    end
    
    
    outfile:write(string.format("\t\t}\n"))
    
    countConv = 1
    countPool = 1
    countPad = 1
    countGroup = 1
    local prevW = "inputW" 
    local prevH = "inputH"
    
    outfile:write(string.format("\t\telse{\n"))
     for i = top, bottom do
        if(i == top) then
           outfile:write(string.format("\t\t\tinputW = inputX;\n"))
           outfile:write(string.format("\t\t\tinputH = strideY + f1K-f1S;\n"))
        end
        
        if(Network.modules[i].__typename == 'nn.SpatialConvolution') then
            outfile:write(string.format("\n\t\t\tinConv%dW = %s;\n", countConv, prevW))
            outfile:write(string.format("\t\t\tinConv%dH = %s;\n", countConv, prevH))
            outfile:write(string.format("\t\t\toutConv%dW = (inConv%dW - f%dK)/f%dS + 1;\n", countConv, countConv, countConv, countConv))
            outfile:write(string.format("\t\t\toutConv%dH = (inConv%dH - f%dK)/f%dS + 1;\n", countConv, countConv, countConv, countConv))
            prevW = string.format("outConv%dW", countConv)
            prevH = string.format("outConv%dH", countConv)
            countConv = countConv + 1 
        else if(Network.modules[i].__typename == 'nn.SpatialMaxPooling') then 
            outfile:write(string.format("\n\t\t\tinPool%dW = %s;\n", countPool, prevW))
            outfile:write(string.format("\t\t\tinPool%dH = %s;\n", countPool, prevH))
            outfile:write(string.format("\t\t\toutPool%dW = (inPool%dW - pool%dK)/pool%dS + 1;\n", countPool, countPool, countPool, countPool))
            outfile:write(string.format("\t\t\toutPool%dH = (inPool%dH - pool%dK)/pool%dS + 1;\n", countPool, countPool, countPool, countPool))
            prevW = string.format("outPad%dW", countPool)
            prevH = string.format("outPad%dH", countPool)
            countPool = countPool + 1 
        else if(Network.modules[i].__typename == 'nn.SpatialZeroPadding') then
             if(i > Group[countGroup]['layer']) then
                countGroup = countGroup + 1
            end 
            outfile:write(string.format("\n\t\t\tpad%d_l = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_r = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_t = 0;\n", countPad))
            outfile:write(string.format("\t\t\tpad%d_b = 0;\n", countPad))

            outfile:write(string.format("\t\t\tif(row == Group%dcond-1)\n", countGroup))
            outfile:write(string.format("\t\t\t\tpad%d_b = pad%d;\n", countPad, countPad))
            outfile:write(string.format("\t\t\telse if(col == Group%dcond-1)\n", countGroup))
            outfile:write(string.format("\t\t\t\tpad%d_r = 0;\n", countPad))            
            outfile:write(string.format("\t\t\telse if(col == Group%dcond-1 && row == Group%dcond-1)\n", countGroup, countGroup))
            outfile:write(string.format("\t\t\t{\n\t\t\t\tpad%d_r = pad%d;\n", countPad, countPad))
            outfile:write(string.format("\t\t\t\tpad%d_b = pad%d;\n\t\t\t}\n", countPad, countPad))
                        
            
            
            outfile:write(string.format("\t\t\tinPad%dW = %s;\n", countPad, prevW))
            outfile:write(string.format("\t\t\tinPad%dH = %s;\n", countPad, prevH))
            outfile:write(string.format("\t\t\toutPad%dW = inPad%dW + pad%d_l + pad%d_r;\n", countPad, countPad, countPad, countPad))
            outfile:write(string.format("\t\t\toutPad%dH = inPad%dH + pad%d_t + pad%d_b;\n", countPad, countPad, countPad, countPad))
            prevW = string.format("outPad%dW", countPad)
            prevH = string.format("outPad%dH", countPad)
            countPad = countPad + 1
              
        else
        
        end 
        end
        end
    end
    
    outfile:write(string.format("\t\t}\n"))
    
    
    ------------------------------------------------------------------------
    
    
    ----------------------------------------------Layers code------------------------------------------------------------     
    for i = top, bottom do
      --For starting group 
      if(Ngroup > 0 and ifcond) then
        outfile:write(string.format("\t\tif(Group%d){\n", GN))
        GN = GN + 1
        Ngroup = Ngroup -1
        ifcond = false    
      end
      
      if(i == top) then
         outfile:write(string.format("\t\t\t<>ReadInput();\n"))
      end 
      
      if(Network.modules[i].__typename == 'nn.SpatialConvolution') then
        if(countConv > 1) then
           outfile:write(string.format("\t\t\t<>ReadData();\n"))          
        end      
        outfile:write(string.format("\t\t\t<>convolution();\n"))
        countConv = countConv + 1
      else if(Network.modules[i].__typename == 'nn.SpatialMaxPooling') then
        if((Network.modules[i].kW - Network.modules[i].dW) > 0) then
          outfile:write(string.format("\t\t\t<>ReadData();\n"))
        end
        outfile:write(string.format("\t\t\t<>pooling();\n")) 
      else if(Network.modules[i].__typename == 'nn.SpatialZeroPadding') then
        outfile:write(string.format("\t\t\t<>padding();\n"))
      end
      end
      end
     
      --For closing group  
      if(i == Group[GN-1]['layer']) then
        outfile:write(string.format("\t\t}\n"))
        ifcond =  true
      end
   end
   
   outfile:write(string.format("}\n"))
   outfile:write(string.format("}\n"))
   
    
    outfile:write(string.format("\n\n"))
end

function FPGAPrintParameters(Network, str, outfile)

    outfile:write(string.format("Pyramid  "))
    for i = 1, #str-1 do
      outfile:write(string.format("%d-", str[i]))  
    end
    outfile:write(string.format("%d", str[#str]))
    outfile:write(string.format("\n\n"))
   
    local pyramid = {}
    for i = #str, 1, -1 do
       pyramid[#pyramid+1] = str[i]
    end
    
 
    for l = 1, #pyramid-1 do
       local top = pyramid[l]+1
       local bottom = pyramid[l+1]
       FPGAParameters(Network, bottom, top, outfile)
    end
    
end

