function [ub,lb,IntCon]=bounds(var_index)
ub = [];
lb = [];
IntCon = [];
varNames = {fieldnames(var_index)};
varNames = varNames{1,1};
numVar = numel(varNames);
for i=1:numVar 
    name = char(varNames(i));
    if name(1) == 'z' || name(1) == 'y' || name(1) == 'w'
        ub = [ub; 1];
        lb = [lb; 0];
%         IntCon = [IntCon; i];
    else
        ub = [ub; inf];
        lb = [lb; 0];
    end
end
    